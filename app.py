from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import whisper
import os
import unicodedata
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np

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

    # Convertir el archivo a WAV si no es WAV
    if audio_file.filename.split('.')[-1] != 'wav':
        try:
            audio = AudioSegment.from_file(file_path)
            file_path = "converted_audio.wav"
            audio.export(file_path, format="wav")
        except Exception as e:
            return jsonify({"error": f"Error al convertir el archivo: {str(e)}"}), 500

    # Recortar el audio al primer minuto (60 segundos)
    try:
        audio = AudioSegment.from_wav(file_path)
        audio = audio[:60 * 1000]  # Recortar al primer minuto
        audio.export(file_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Error al recortar el audio: {str(e)}"}), 500

    try:
        # Diarización
        [flags, classes, centers] = aS.speaker_diarization(file_path, n_speakers=3)
        classes = np.array(classes)

        # Verificar que el array sea 1D y aplanarlo si es necesario
        if classes.ndim == 0:
            return jsonify({"error": "No se detectaron segmentos de habla para diarización."}), 500
        elif classes.ndim > 1:
            classes = classes.flatten()

        # Asegurarse de que todas las clases sean numéricas
        try:
            classes = classes.astype(int)
        except ValueError:
            return jsonify({"error": "La lista de clases contiene elementos no numéricos o inconsistentes."}), 500

        segments = []
        if len(classes) > 0:
            current_speaker = int(classes[0])
            start_time = 0

            # Identificar los segmentos de cada hablante
            for i, speaker in enumerate(classes):
                if speaker != current_speaker:
                    segments.append((start_time, i, current_speaker))
                    start_time = i
                    current_speaker = speaker

            segments.append((start_time, len(classes), current_speaker))
        else:
            return jsonify({"error": "No se detectaron segmentos de habla para diarización."}), 500

    except Exception as e:
        return jsonify({"error": f"Error al realizar la diarización: {str(e)}"}), 500

    # Cargar el modelo de Whisper
    model = whisper.load_model("medium")

    results = []

    # Calcular la duración de cada frame
    frame_rate = len(audio) / len(classes)

    # Procesar cada segmento para transcripción
    for idx, (start_frame, end_frame, speaker) in enumerate(segments):
        start_time = int(start_frame * frame_rate)
        end_time = int(end_frame * frame_rate)
        segment_audio = audio[start_time:end_time]

        segment_path = f"segment_{idx}.wav"
        segment_audio.export(segment_path, format="wav")

        try:
            # Transcribir el segmento
            result = model.transcribe(segment_path, language="es")
            text = result["text"]
            text = unicodedata.normalize("NFKC", text)

            results.append({
                "speaker": speaker,
                "text": text
            })
        except Exception as e:
            results.append({
                "speaker": speaker,
                "error": f"Error al transcribir el segmento: {str(e)}"
            })

        # Eliminar el archivo temporal del segmento
        os.remove(segment_path)

    # Devolver los resultados
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)