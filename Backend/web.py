import os
from pathlib import Path
from flask import Flask, render_template, request

try:
    # When running as a package (gunicorn Backend.web:app)
    from .model import chatbot_response
except ImportError:
    # When running as a script (python Backend/web.py)
    from model import chatbot_response

# Resolve project root so templates and static assets load correctly
PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "template"),
    static_folder=str(PROJECT_ROOT / "static"),
)

@app.route("/")
def home():
    return render_template('Home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route("/get")
def get_bot_response():
    userinput = request.args.get("text")
    return str(chatbot_response(userinput))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)