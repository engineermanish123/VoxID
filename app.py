from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import spacy
from spellchecker import SpellChecker
import re
from tempfile import NamedTemporaryFile
import os
import requests
from urllib.parse import urlparse
from pydub import AudioSegment
import openai
import fasttext
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import json
from collections import OrderedDict
from langdetect import detect
from deep_translator import GoogleTranslator

load_dotenv()

openai.api_key = os.getenv("TRANSCRIPTION_API")
nlp = spacy.load('en_core_web_sm')
spell = SpellChecker()

model_path = 'Sample//lid.176.bin'
fasttext_model = fasttext.load_model(model_path)

app = Flask(__name__)
CORS(app)

# Language mappings
LANGUAGE_MAP = {
    "en": "English", "fr": "French", "hi": "Hindi", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh-cn": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "tr": "Turkish", "nl": "Dutch", "el": "Greek",
    "sv": "Swedish", "pl": "Polish", "iw": "Hebrew", "bn": "Bengali", "th": "Thai",
    "id": "Indonesian", "vi": "Vietnamese", "ro": "Romanian", "fa": "Persian", "uk": "Ukrainian",
    "ur": "Urdu", "ta": "Tamil", "te": "Telugu", "ms": "Malay", "hu": "Hungarian"
}
language_mapping = LANGUAGE_MAP  # For fastText mapping reuse

def extract_filename_from_url(url):
    parsed_url = urlparse(url)
    file_name = re.sub(r'[^\w\s-]', '', os.path.basename(parsed_url.path))
    file_name = re.sub(r'[-\s]+', '_', file_name).strip('_')
    return file_name

def diarize_audio(file_path):
    try:
        print(f"Processing file: {file_path}")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("USER_AUTH_TOKEN"))
        diarization = pipeline({'uri': 'filename', 'audio': file_path})
        return diarization
    except Exception as e:
        print(f"Error during diarization: {e}")
        return None

def split_audio_by_speaker(file_path, diarization):
    audio = AudioSegment.from_wav(file_path)
    chunks = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = int(turn.start * 1000)
        end_time = int(turn.end * 1000)
        chunk = audio[start_time:end_time]
        chunks.append((chunk, speaker, turn.start))
    return chunks

def transcribe_chunk(chunk):
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            chunk.export(temp_wav.name, format="wav")
            with open(temp_wav.name, 'rb') as audio_file:
                response = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="json")
                transcription = response.get('text', '')
            os.remove(temp_wav.name)
        return transcription
    except Exception as e:
        print(f"Error during chunk transcription: {e}")
        return ""

def detect_language_with_fasttext(text):
    cleaned_text = text.replace('\n', ' ').strip()
    if cleaned_text:
        predictions = fasttext_model.predict(cleaned_text)
        detected_language_code = predictions[0][0].replace('__label__', '')
        detected_language_name = language_mapping.get(detected_language_code, 'Unknown Language')
        return f"{detected_language_name} ({detected_language_code})"
    else:
        return "Unknown Language (empty text)"

def translate_text(text):
    translation_response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who translates text."},
            {"role": "user", "content": f"Translate the following text to English:\n\n{text}"}
        ],
        max_tokens=1000,
        temperature=0.1
    )
    return translation_response['choices'][0]['message']['content'].strip()

def process_and_print_chunk(chunk_speaker_time, speaker_mapping):
    chunk, speaker, start_time = chunk_speaker_time
    transcription = transcribe_chunk(chunk)
    caller_label = speaker_mapping.get(speaker, f"Unknown Speaker ({speaker})")
    timestamp = f"{int(start_time // 60):02}:{int(start_time % 60):02}"
    return f"{timestamp} {caller_label}: {transcription}"

def download_audio_from_url(url, file_name):
    response = requests.get(url)
    response.raise_for_status()
    file_path = f'tmp/{file_name}.wav'
    with open(file_path, 'wb') as f:
        f.write(response.content)
    return file_path

def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in LANGUAGE_MAP else "en"
    except:
        return "en"

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    message = data.get('message')
    target_language = 'en'
    if not message:
        return jsonify({'error': 'Message is required'}), 400

    source_lang = detect_language(message)
    source_lang_name = LANGUAGE_MAP.get(source_lang, "Unknown")
    try:
        translated = GoogleTranslator(source=source_lang, target=target_language).translate(message)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "original_text": message,
        "detected_language": source_lang,
        "detected_language_name": source_lang_name,
        "translated_text": translated,
        "target_language": target_language
    })

@app.route('/spellcheck', methods=['POST'])
def check_spelling():
    data = request.get_json()
    text = data.get('text', '')
    doc = nlp(text)
    misspelled_words = [{'word': token.text} for token in doc if not token.is_punct and token.text.lower() not in spell]
    return jsonify({'misspelled': misspelled_words})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    authToken = request.headers.get('Authorization')
    if authToken != os.getenv("API_TOKEN"):
        return jsonify({"error": "Invalid Request"}), 403

    file = request.files.get('file')
    url = request.form.get('url')
    if not file and not url:
        return jsonify({"error": "No file or URL provided"}), 400

    if file:
        file_name = re.sub(r'\W+', '_', file.filename.rsplit('.', 1)[0])
        file_path = f"tmp/{file_name}.wav"
        file.save(file_path)
    elif url:
        file_name = extract_filename_from_url(url)
        file_path = download_audio_from_url(url, file_name)

    json_file_path = f"tmp/{file_name}.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            saved_response = json.load(file, object_pairs_hook=OrderedDict)
        return Response(json.dumps(saved_response, indent=4), mimetype='application/json')

    try:
        diarization = diarize_audio(file_path)
        if not diarization:
            return jsonify({"error": "Diarization failed."}), 500

        audio_chunks = split_audio_by_speaker(file_path, diarization)
        speakers = sorted(set(speaker for _, speaker, _ in audio_chunks))
        speaker_mapping = {speaker: f"Caller {i+1}" for i, speaker in enumerate(speakers)}

        full_transcription = ""
        all_transcriptions = []
        for chunk_speaker_time in audio_chunks:
            formatted_output = process_and_print_chunk(chunk_speaker_time, speaker_mapping)
            full_transcription += f"{formatted_output} "
            all_transcriptions.append(formatted_output)

        detected_language = detect_language_with_fasttext(full_transcription)
        translated_text = ""
        if 'en' not in detected_language:
            translated_text = translate_text(full_transcription)

        response_data = OrderedDict([
            ("converted_transcription", "\n".join(all_transcriptions)),
            ("original_language", detected_language),
            ("original_transcription", "\n".join(all_transcriptions)),
            ("converted_language", "English (en)"),
            ("transcription", translated_text if translated_text else "\n".join(all_transcriptions))
        ])

        with open(json_file_path, 'w') as json_file:
            json.dump(response_data, json_file, indent=4)

        return Response(json.dumps(response_data, indent=4), mimetype='application/json')
    # except Exception as e:
    #     return jsonify({"error": "An error occurred during transcription"}), 500
    except Exception as e:
        print("Exception occurred:", str(e))  # Console me error print hoga
        return jsonify({"error": "An error occurred during transcription", "details": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8962, debug=False)





