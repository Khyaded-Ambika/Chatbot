import json
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


def ensure_nltk_resources() -> None:
    for pkg in ["punkt", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def build_training_data(intents_json: dict, lemmatizer: WordNetLemmatizer):
    words: list[str] = []
    classes: list[str] = []
    documents: list[tuple[list[str], str]] = []
    ignore_tokens = {"?", "!", "*", "(", ")", "&"}

    for intent in intents_json.get("intents", []):
        tag = intent.get("tag", "").strip()
        if not tag:
            continue
        for pattern in intent.get("patterns", []):
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))
            if tag not in classes:
                classes.append(tag)

    words = [lemmatizer.lemmatize(token.lower()) for token in words if token not in ignore_tokens]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    training: list[list[list[int] | list[int]]] = []
    output_empty = [0] * len(classes)

    for tokens, tag in documents:
        bag: list[int] = []
        token_words = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        for w in words:
            bag.append(1 if w in token_words else 0)
        output_row = list(output_empty)
        output_row[classes.index(tag)] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    return words, classes, train_x, train_y


def train_and_save(data_dir: Path, epochs: int = 200, batch_size: int = 5) -> None:
    ensure_nltk_resources()
    lemmatizer = WordNetLemmatizer()

    intents_path = data_dir / "intents.json"
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_json = json.load(f)

    words, classes, train_x, train_y = build_training_data(intents_json, lemmatizer)

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))

    optimizer = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)

    # Save artifacts
    model.save(str(data_dir / "model.h5"))
    with open(data_dir / "words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open(data_dir / "classes.pkl", "wb") as f:
        pickle.dump(classes, f)
    with open(data_dir / "trainHistory.pkl", "wb") as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_directory = project_root / "data"
    print(f"Training using intents at: {data_directory / 'intents.json'}")
    train_and_save(data_directory)
    print("Training complete. Artifacts saved to data/.")


