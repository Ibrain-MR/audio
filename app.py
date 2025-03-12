from flask import Flask, render_template, request, jsonify, Response
from pydub import AudioSegment
import whisper
import os
import unicodedata

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = "uploaded_audio." + audio_file.filename.split('.')[-1]
    audio_file.save(file_path)

    if audio_file.filename.split('.')[-1] != 'wav':
        try:
            audio = AudioSegment.from_file(file_path)
            file_path = "converted_audio.wav"
            audio.export(file_path, format="wav")
        except Exception as e:
            return jsonify({"error": f"Error al convertir el archivo: {str(e)}"}), 500

    model = whisper.load_model("medium")

    try:
        def generate_transcription():
            result = model.transcribe(file_path, language="es")
            text = result["text"]
            text = unicodedata.normalize("NFKC", text)
            for word in text.split():
                yield word + " "
                import time; time.sleep(0.1)  # Simula un pequeño retraso para la sensación progresiva

        return Response(generate_transcription(), content_type='text/plain; charset=utf-8')
    
    except Exception as e:
        return jsonify({"error": f"Error al transcribir el audio: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)