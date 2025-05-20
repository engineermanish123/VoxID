from flask import Flask, request, jsonify
import openai
import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import requests

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("TRANSCRIPTION_API")

def diarize_audio(file_path):
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("USE_AUTH_TOKEN"))
        
        if pipeline is None:
            raise Exception("Failed to load the model.")
        
        diarization = pipeline(file_path)
        print("Diarization completed successfully.")
        return diarization
    
    except Exception as e:
        print(f"Error during diarization: {e}")
        return None

def split_audio_by_speaker(file_path, diarization):
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = int(turn.start * 1000)  # Convert to milliseconds
        end_time = int(turn.end * 1000)  # Convert to milliseconds
        chunk = audio[start_time:end_time]
        chunks.append((chunk, speaker))
    return chunks

def transcribe_chunk(chunk):
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            chunk.export(temp_wav.name, format="wav")
            temp_wav.close()  # Close the file so it can be opened by another process
            
            with open(temp_wav.name, 'rb') as audio_file:
                response = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="json")
                transcription = response.get('text', '')
            os.remove(temp_wav.name)  # Clean up the temporary file after use
        return transcription
    
    except Exception as e:
        print(f"Error during chunk transcription: {e}")
        return ""

def process_chunk(chunk_speaker, speaker_mapping):
    chunk, speaker = chunk_speaker
    transcription = transcribe_chunk(chunk)
    caller_label = speaker_mapping.get(speaker, f"Unknown Speaker ({speaker})")
    return f"{caller_label}: {transcription.strip()}"

def download_audio_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        print(f"Error downloading audio from URL: {e}")
        return None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files and 'url' not in request.form:
        return jsonify({"error": "No file or URL provided"}), 400
    
    file = request.files.get('file')
    url = request.form.get('url')
    
    temp_file_path = None

    if file:
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not file.filename.endswith('.wav'):
            return jsonify({"error": "Only .wav files are allowed"}), 400
        
        try:
            # Save the uploaded file to a temporary location
            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elif url:
        if not url.startswith("http"):
            return jsonify({"error": "Invalid URL"}), 400
        
        try:
            temp_file_path = download_audio_from_url(url)
            if not temp_file_path:
                return jsonify({"error": "Failed to download audio from URL"}), 500
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if not temp_file_path:
        return jsonify({"error": "Failed to obtain audio file"}), 500
    
    try:
        # Diarize and process the audio
        diarization = diarize_audio(temp_file_path)
        
        if not diarization:
            return jsonify({"error": "Diarization failed"}), 500
        
        audio_chunks = split_audio_by_speaker(temp_file_path, diarization)
        speakers = sorted(set(speaker for _, speaker in audio_chunks))
        speaker_mapping = {speaker: f"Caller {i+1}" for i, speaker in enumerate(speakers)}
        
        with ThreadPoolExecutor() as executor:
            formatted_transcriptions = list(executor.map(lambda chunk_speaker: process_chunk(chunk_speaker, speaker_mapping), audio_chunks))
        
        # Join the transcriptions with newline characters
        formatted_transcription_text = "\n".join(formatted_transcriptions).strip()
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return jsonify({"transcription": formatted_transcription_text}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8962)
