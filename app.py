from flask import Flask, request, jsonify
import tempfile
import os
from pydub import AudioSegment
from speechbrain.inference.classifiers import EncoderClassifier

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="emotion_model"
)

EMOTION_MAP = {
    "ang": "raiva 😡",
    "hap": "feliz 😄",
    "sad": "triste 😢",
    "neu": "neutro 😐"
}


def convert_to_wav(input_path):
    wav_path = input_path + ".wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(wav_path, format="wav")

    return wav_path


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "voice-emotion-api"
    })


@app.route("/emotion", methods=["POST"])
def emotion():

    temp_input = None
    temp_wav = None

    try:

        if "audio" not in request.files:
            return jsonify({
                "error": "Campo 'audio' não enviado"
            }), 400

        audio_file = request.files["audio"]

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            audio_file.save(tmp.name)
            temp_input = tmp.name

        temp_wav = convert_to_wav(temp_input)

        out_prob, score, index, text_lab = classifier.classify_file(temp_wav)

        raw_emotion = text_lab[0]

        emotion = EMOTION_MAP.get(
            raw_emotion.lower(),
            raw_emotion
        )

        confidence = float(score)

        return jsonify({
            "emotion": emotion,
            "raw_label": raw_emotion,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

    finally:

        try:
            if temp_input and os.path.exists(temp_input):
                os.remove(temp_input)

            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)

        except:
            pass


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )
