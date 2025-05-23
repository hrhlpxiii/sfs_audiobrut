from flask import Flask, request, jsonify
import librosa
import tempfile
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "API d'analyse audio brute SONG FOR SUCCESS est active.",
        "endpoint": "/analyse (POST audio)"
    }), 200

@app.route("/analyse", methods=["POST"])
def analyse():
    if 'audio' not in request.files:
        return jsonify({"error": "Fichier audio manquant dans la requête."}), 400

    try:
        file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            file.save(tmp.name)

            # Analyse audio avec librosa
            # Chargement optimisé : 20s max, mono, 22050 Hz (plus rapide)
            y, sr = librosa.load(tmp.name, sr=22050, mono=True, duration=20.0)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
            key_index = chroma.argmax()
            key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][key_index]
            energy = float(librosa.feature.rms(y=y).mean().item())

        os.unlink(tmp.name)

        return jsonify({
            "tempo": round(float(tempo), 2),
            "key": key,
            "energy": round(float(energy), 5)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
