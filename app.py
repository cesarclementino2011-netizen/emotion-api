from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "online"
    })

@app.route("/emotion")
def emotion():
    return jsonify({
        "emotion": "happy",
        "confidence": 0.95
    })
