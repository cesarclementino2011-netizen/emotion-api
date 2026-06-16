from flask import Flask, request, jsonify
import tempfile
import os
import whisper
from transformers import pipeline
from pydub import AudioSegment

app = Flask(__name__)

# segurança básica
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# modelos carregados uma vez
whisper_model = whisper.load_model("base")

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# emoções finais
EMOTION_MAP = {
    "joy": "feliz 😄",
    "anger": "raiva 😡",
    "sadness": "triste 😢",
    "fear": "medo 😨",
    "surprise": "surpreso 😲",
    "neutral": "neutro 😐",
    "love": "amor ❤️"
}


def convert_to_wav(input_path):
    audio = AudioSegment.from_file(input_path)
    wav_path = input_path + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "emotion-api"
    })


@app.route("/emotion", methods=["POST"])
def emotion():

    tmp_path = None
    wav_path = None

    try:
        if "audio" not in request.files:
            return jsonify({"error": "audio missing"}), 400

        audio = request.files["audio"]

        # salva qualquer formato (ogg/mp3/wav)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            audio.save(tmp.name)
            tmp_path = tmp.name

        # converte para wav
        wav_path = convert_to_wav(tmp_path)

        # áudio → texto
        result = whisper_model.transcribe(wav_path, language="pt")
        text = result["text"].strip()

        # emoção no texto
        emotion_result = emotion_model(text)[0]

        label = emotion_result["label"]
        score = float(emotion_result["score"])

        emotion = EMOTION_MAP.get(label, label)

        return jsonify({
            "transcription": text,
            "emotion": emotion,
            "raw_label": label,
            "confidence": round(score, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
