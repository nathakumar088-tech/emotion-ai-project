from flask import Flask, render_template, request
import os
import subprocess
import numpy as np
import librosa
import onnxruntime as ort
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# ==============================
# UPLOAD FOLDER
# ==============================

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
# AI EMOTION MODEL
# ==============================

MODEL_REPO = "onnx-community/Speech-Emotion-Classification-ONNX"
MODEL_FILE = "onnx/model_int8.onnx"

emotion_session = None


def get_emotion_session():

    global emotion_session

    if emotion_session is None:

        print("Downloading/loading ONNX emotion model...")

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE
        )

        emotion_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        print("Emotion ONNX model loaded successfully!")

    return emotion_session


# ==============================
# EMOTION LABELS
# ==============================

emotion_names = {
    "ANG": "Angry",
    "CAL": "Calm",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
    "SUR": "Surprised"
}


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template("index.html")


# ==============================
# ANALYSIS PAGE
# ==============================

@app.route("/analysis")
def analysis():

    return render_template("analysis.html")


# ==============================
# EMOTION-AWARE CHATBOT
# ==============================

@app.route("/chatbot", methods=["POST"])
def chatbot():

    message = request.form.get("message", "").strip()

    if not message:
        return "Please enter a message."

    text = message.lower()

    if any(
        word in text
        for word in ["happy", "good", "great", "excited"]
    ):

        reply = (
            "That's nice to hear! 😊 "
            "What made you feel this way?"
        )

    elif any(
        word in text
        for word in ["sad", "upset", "bad", "lonely"]
    ):

        reply = (
            "I'm sorry you're having a difficult moment. "
            "You can talk about what's bothering you."
        )

    elif any(
        word in text
        for word in ["angry", "mad", "frustrated"]
    ):

        reply = (
            "It sounds like something is frustrating you. "
            "Taking a short pause and talking about it may help."
        )

    elif any(
        word in text
        for word in ["hello", "hi", "hey"]
    ):

        reply = (
            "Hello! 👋 "
            "I'm your EmotionAI assistant. "
            "How are you feeling today?"
        )

    else:

        reply = (
            "I understand. "
            "Tell me a little more about how you're feeling."
        )

    return reply


# ==============================
# CONVERT WEBM/OTHER AUDIO TO WAV
# ==============================

def convert_to_wav(input_path):

    base_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    output_path = os.path.join(
        UPLOAD_FOLDER,
        base_name + "_converted.wav"
    )

    print("Converting audio to WAV...")
    print("Input:", input_path)
    print("Output:", output_path)

    try:

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                output_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    except FileNotFoundError:

        raise RuntimeError(
            "FFmpeg is not installed on the server."
        )

    except subprocess.CalledProcessError as e:

        error_message = e.stderr.decode(
            "utf-8",
            errors="ignore"
        )

        print("FFmpeg Error:")
        print(error_message)

        raise RuntimeError(
            "Unable to convert the recorded audio."
        )

    if not os.path.exists(output_path):

        raise RuntimeError(
            "WAV conversion failed."
        )

    print("WAV conversion successful!")

    return output_path


# ==============================
# AUDIO PREPROCESSING
# ==============================

def prepare_audio(file_path):

    print("Loading WAV audio...")

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    audio = audio.astype(np.float32)

    # Remove leading/trailing silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    # Maximum 10 seconds
    max_samples = 16000 * 10

    if len(audio) > max_samples:
        audio = audio[:max_samples]

    if len(audio) == 0:
        raise ValueError(
            "Audio file contains no usable speech."
        )

    # Wav2Vec2 feature extraction uses input normalization.
    mean = np.mean(audio)
    std = np.std(audio)

    if std > 1e-7:
        audio = (audio - mean) / std

    audio = audio.astype(np.float32)

    print("Audio prepared successfully!")
    print("Sample rate:", sample_rate)
    print("Audio samples:", len(audio))
    print("Audio mean:", float(np.mean(audio)))
    print("Audio std:", float(np.std(audio)))

    return audio


# ==============================
# AI PREDICTION
# ==============================

def predict_emotion(file_path):

    session = get_emotion_session()

    audio = prepare_audio(file_path)

    input_name = session.get_inputs()[0].name

    print("ONNX input name:", input_name)
    print("ONNX input shape:", session.get_inputs()[0].shape)

    input_data = np.expand_dims(
        audio,
        axis=0
    ).astype(np.float32)

    outputs = session.run(
        None,
        {
            input_name: input_data
        }
    )

    print("ONNX prediction completed.")

    logits = np.asarray(outputs[0])

    # Remove batch / extra dimensions.
    logits = np.squeeze(logits)

    print("Raw model output:", logits)

    # This model must return 8 emotion scores.
    if logits.size != 8:
        raise RuntimeError(
            f"Unexpected model output size: {logits.size}. "
            "Expected 8 emotion scores."
        )

    logits = logits.astype(np.float64)

    # Stable softmax.
    logits = logits - np.max(logits)

    exp_values = np.exp(logits)

    probabilities = (
        exp_values /
        np.sum(exp_values)
    )

    # Official model label order.
    labels = [
        "ANG",
        "CAL",
        "DIS",
        "FEA",
        "HAP",
        "NEU",
        "SAD",
        "SUR"
    ]

    results = []

    for label, probability in zip(
        labels,
        probabilities
    ):

        results.append(
            {
                "label": label,
                "score": float(probability)
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("Emotion results:")

    for result in results:
        print(
            result["label"],
            round(result["score"] * 100, 2),
            "%"
        )

    return results


# ==============================
# AUDIO UPLOAD + AI ANALYSIS
# ==============================

@app.route("/upload", methods=["POST"])
def upload():

    print("================================")
    print("UPLOAD REQUEST RECEIVED")
    print("================================")

    if "audio" not in request.files:

        return "No audio file selected"

    audio = request.files["audio"]

    if audio.filename == "":

        return "No audio file selected"

    original_filename = audio.filename

    # Make safe filename
    safe_filename = os.path.basename(
        original_filename
    )

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        safe_filename
    )

    audio.save(file_path)

    print("Audio saved:", file_path)

    converted_path = None

    try:

        # ==================================
        # WEBM / OGG / MP3 -> WAV
        # ==================================

        converted_path = convert_to_wav(
            file_path
        )

        print(
            "Converted audio:",
            converted_path
        )

        # ==================================
        # AI ANALYSIS
        # ==================================

        results = predict_emotion(
            converted_path
        )

    except Exception as e:

        print("================================")
        print("AI ERROR")
        print("================================")
        print(str(e))
        print("================================")

        return (
            f"Error analyzing audio: {str(e)}"
        )

    # ==============================
    # EMOTION RESULTS
    # ==============================

    emotion_html = ""

    for result in results:

        label = result["label"]

        score = result["score"] * 100

        emotion = emotion_names.get(
            label,
            label
        )

        emotion_html += f"""
        <div class="emotion-card">

            <div class="emotion-header">

                <span class="emotion-name">
                    {emotion}
                </span>

                <span class="emotion-score">
                    {score:.2f}%
                </span>

            </div>

            <div class="bar">

                <div
                    class="fill"
                    style="width: {score}%">
                </div>

            </div>

        </div>
        """

    # ==============================
    # TOP EMOTION
    # ==============================

    top_label = results[0]["label"]

    top_emotion = emotion_names.get(
        top_label,
        top_label
    )

    top_score = results[0]["score"] * 100

    # ==============================
    # RESULT PAGE
    # ==============================

    return f"""
<!DOCTYPE html>

<html>

<head>

    <title>Emotion Analysis</title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0">

    <style>

        * {{
            box-sizing: border-box;
        }}

        body {{

            margin: 0;

            padding: 20px;

            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #eef2ff,
                    #f5f3ff
                );

            min-height: 100vh;
        }}

        .container {{

            max-width: 700px;

            margin: 40px auto;
        }}

        .result-box {{

            background: white;

            padding: 35px;

            border-radius: 25px;

            box-shadow:
                0 15px 40px
                rgba(0,0,0,0.12);
        }}

        h1 {{

            text-align: center;

            color: #312e81;

            margin-bottom: 10px;
        }}

        .audio-name {{

            text-align: center;

            color: #64748b;

            margin-bottom: 30px;
        }}

        .main-result {{

            text-align: center;

            background: #eef2ff;

            padding: 30px;

            border-radius: 20px;

            margin-bottom: 30px;
        }}

        .main-emotion {{

            font-size: 42px;

            font-weight: bold;

            color: #4f46e5;
        }}

        .main-score {{

            font-size: 20px;

            color: #475569;

            margin-top: 10px;
        }}

        h2 {{

            color: #1e293b;

            margin-bottom: 20px;
        }}

        .emotion-card {{

            margin-bottom: 22px;
        }}

        .emotion-header {{

            display: flex;

            justify-content: space-between;

            margin-bottom: 8px;
        }}

        .emotion-name {{

            font-size: 18px;

            font-weight: bold;

            color: #1e293b;
        }}

        .emotion-score {{

            color: #475569;

            font-weight: bold;
        }}

        .bar {{

            width: 100%;

            height: 15px;

            background: #e2e8f0;

            border-radius: 20px;

            overflow: hidden;
        }}

        .fill {{

            height: 100%;

            background:
                linear-gradient(
                    90deg,
                    #6366f1,
                    #8b5cf6
                );

            border-radius: 20px;
        }}

        .back {{

            display: block;

            width: fit-content;

            margin: 35px auto 0;

            padding: 14px 25px;

            background: #4f46e5;

            color: white;

            text-decoration: none;

            border-radius: 12px;

            font-weight: bold;
        }}

        .back:hover {{

            background: #3730a3;
        }}

        @media (max-width: 600px) {{

            body {{

                padding: 12px;
            }}

            .container {{

                margin: 15px auto;
            }}

            .result-box {{

                padding: 22px;
            }}

            h1 {{

                font-size: 27px;
            }}

            .main-emotion {{

                font-size: 34px;
            }}

        }}

    </style>

</head>

<body>

    <div class="container">

        <div class="result-box">

            <h1>
                🎙️ Emotion Analysis
            </h1>

            <div class="audio-name">

                Audio:

                <strong>
                    {safe_filename}
                </strong>

            </div>

            <div class="main-result">

                <div class="main-emotion">
                    {top_emotion}
                </div>

                <div class="main-score">
                    Confidence: {top_score:.2f}%
                </div>

            </div>

            <h2>
                Emotion Scores
            </h2>

            {emotion_html}

            <a
                href="/"
                class="back">

                🔄 Analyze Another Audio

            </a>

        </div>

    </div>

</body>

</html>
"""


# ==============================
# RUN FLASK
# ==============================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )
