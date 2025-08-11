# Kaira – Interactive University Chatbot

Kaira is a web-based chatbot designed to provide university-related information to students efficiently, especially addressing the challenges of remote access during the COVID-19 pandemic. It uses deep learning techniques to understand and respond to natural language queries.

## Features

- Deep learning-based chatbot using a 3-layer Recurrent Neural Network (RNN) trained on curated intents.
- Backend developed with Flask to process user queries and serve responses in real-time.
- Responsive frontend built with HTML, CSS, JavaScript, and Bootstrap.
- Speech-to-text input support for enhanced accessibility.
- Easily updatable knowledge base to keep information current.

## Technologies Used

- Python, Flask
- Keras, TensorFlow
- NLTK for text preprocessing
- HTML, CSS, JavaScript, Bootstrap
- Web Speech API for speech-to-text

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kaira-chatbot.git
   cd kaira-chatbot
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the Flask app:
   ```bash
   python app.py
5. Open your browser at http://localhost:5000 to interact with Kaira.

## Model Training

- The RNN model is trained on a JSON dataset containing 45 intents with associated patterns and responses.
- Text data is tokenized, lemmatized, and preprocessed using NLTK.
- The model is trained for 500 epochs with Adam optimizer and categorical cross-entropy loss.
- Dropout layers are applied to reduce overfitting.

## Future Enhancements
- Integrate with official university platforms for live data updates.
- Add text-to-speech functionality for better interaction.
- Implement user authentication for personalized experiences.
   
