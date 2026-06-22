import os
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import ml_utils


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


SERVICES = {
    "dbscan": {
        "title": "DBSCAN Clustering",
        "category": "Clustering",
        "description": "Upload a numeric CSV dataset and discover density-based clusters.",
        "input_type": "file",
        "accept": ".csv",
        "sample_hint": "CSV should include at least one numeric column.",
    },
    "kmeans": {
        "title": "K-Means Clustering",
        "category": "Clustering",
        "description": "Upload a numeric CSV dataset and split it into K-Means clusters.",
        "input_type": "file",
        "accept": ".csv",
        "sample_hint": "CSV should include at least one numeric column.",
    },
    "cnn": {
        "title": "Image Classification",
        "category": "Computer Vision",
        "description": "Upload an image and classify it as Male or Female with the dummy CNN.",
        "input_type": "file",
        "accept": "image/*",
        "sample_hint": "Run python create_cnn.py once before using this service.",
    },
    "sentiment_voice": {
        "title": "Voice Sentiment Analysis",
        "category": "Audio NLP",
        "description": "Upload a WAV audio file, transcribe speech, and detect sentiment.",
        "input_type": "file",
        "accept": ".wav,.aiff,.aif,.flac",
        "sample_hint": "SpeechRecognition works best with clear WAV audio.",
    },
    "qa_voice": {
        "title": "Voice Question Answering",
        "category": "Audio NLP",
        "description": "Ask a question by voice and receive the answer as generated audio.",
        "input_type": "file",
        "accept": ".wav,.aiff,.aif,.flac",
        "sample_hint": "The backend uses a hardcoded Applied ML context.",
    },
    "text_gen": {
        "title": "Text Generation",
        "category": "Generative NLP",
        "description": "Enter a prompt and generate a short continuation.",
        "input_type": "text",
        "placeholder": "Machine learning helps students...",
    },
    "translation": {
        "title": "English to Urdu Translation",
        "category": "Translation",
        "description": "Translate English text into Urdu using a Hugging Face model.",
        "input_type": "text",
        "placeholder": "Artificial intelligence is changing education.",
    },
    "ner": {
        "title": "Named Entity Recognition",
        "category": "Information Extraction",
        "description": "Extract people, places, organizations, and other entities from text.",
        "input_type": "text",
        "placeholder": "Abdullahh studies Applied Machine Learning at UCP in Lahore.",
    },
    "apriori": {
        "title": "Apriori Association Rules",
        "category": "Data Mining",
        "description": "Upload a transaction CSV and generate frequent item rules.",
        "input_type": "file",
        "accept": ".csv",
        "sample_hint": "Works with one-hot CSVs or transaction rows like milk,bread,butter.",
    },
}


def save_upload(uploaded_file):
    original_name = secure_filename(uploaded_file.filename)
    filename = f"{uuid4().hex}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(filepath)
    return filepath


@app.route("/")
def index():
    return render_template("index.html", services=SERVICES)


@app.route("/service/<task>", methods=["GET", "POST"])
def service(task):
    service_config = SERVICES.get(task)
    if service_config is None:
        return render_template(
            "service.html",
            service=None,
            task=task,
            error="Unknown service selected.",
        ), 404

    result = None
    error = None

    if request.method == "POST":
        try:
            if service_config["input_type"] == "file":
                uploaded_file = request.files.get("file")
                if not uploaded_file or uploaded_file.filename == "":
                    raise ValueError("Please upload a file before processing.")

                filepath = save_upload(uploaded_file)

                if task == "dbscan":
                    result = ml_utils.run_clustering(filepath, algorithm="dbscan")
                elif task == "kmeans":
                    result = ml_utils.run_clustering(filepath, algorithm="kmeans")
                elif task == "cnn":
                    result = ml_utils.classify_image(filepath)
                elif task == "sentiment_voice":
                    result = ml_utils.sentiment_from_voice(filepath)
                elif task == "qa_voice":
                    result = ml_utils.answer_question_from_voice(filepath)
                    if result.get("audio_file"):
                        result["audio_url"] = url_for(
                            "static",
                            filename=f"uploads/{os.path.basename(result['audio_file'])}",
                        )
                elif task == "apriori":
                    result = ml_utils.run_apriori(filepath)
            else:
                text = request.form.get("text_input", "").strip()
                if not text:
                    raise ValueError("Please enter text before processing.")

                if task == "text_gen":
                    result = ml_utils.generate_text(text)
                elif task == "translation":
                    result = ml_utils.translate_en_to_ur(text)
                elif task == "ner":
                    result = ml_utils.extract_ner(text)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "service.html",
        task=task,
        service=service_config,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
