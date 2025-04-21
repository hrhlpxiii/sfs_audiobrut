from flask import Flask, request, jsonify
import librosa
import tempfile
import os

app = Flask(__name__)

@app.route("/analyse", methods=["POST"])
def analyse():
    if 'audio' not in request.files:
        return jsonify({"error": "Fichier audio manquant dans la requête."}), 400

    try:
        file = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            file.save(tmp.name)

            # Chargement de l'audio
            y, sr = librosa.load(tmp.name)

            # Analyse avec arguments keyword-only (corrigé)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
            key = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][chroma.argmax()]
            energy = librosa.feature.rms(y=y).mean()

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

