from flask import Flask, request, jsonify
import io
import requests
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

app = Flask(__name__)

# Path to your service account key file
client_file = './saferwatchai-29766913ea77.json'

# Create the credentials object
credentials = service_account.Credentials.from_service_account_file(client_file)

# Create a Speech client with the credentials
client = speech.SpeechClient(credentials=credentials)

def download_audio_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return io.BytesIO(response.content)  # Use BytesIO to handle the in-memory file
    except Exception as e:
        print(f"Error downloading audio from URL: {e}")
        return None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file = request.files.get('file')
    url = request.form.get('url')

    if not audio_file and not url:
        return jsonify({'error': 'No file or URL provided'}), 400

    if audio_file:
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
    elif url:
        if not url.startswith("http"):
            return jsonify({'error': 'Invalid URL'}), 400
        audio_content = download_audio_from_url(url)
        if not audio_content:
            return jsonify({'error': 'Failed to download audio from URL'}), 500
        audio = speech.RecognitionAudio(content=audio_content.read())

    # Configure the request for transcription with speaker diarization
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,  # Set this to match your audio file's sample rate
        language_code='en-US',
        enable_speaker_diarization=True
    )

    # Use long_running_recognize for longer audio files
    operation = client.long_running_recognize(config=config, audio=audio)

    print('Processing audio...')

    # Wait for the operation to complete
    response = operation.result(timeout=300)

    # Determine the number of speakers
    speaker_ids = set()
    for result in response.results:
        for word_info in result.alternatives[0].words:
            speaker_ids.add(word_info.speaker_tag)

    speaker_labels = {speaker_id: f"Speaker {i + 1}" for i, speaker_id in enumerate(speaker_ids)}

    # Process and prepare the results with speaker labels
    sentences = []
    current_sentence = []
    current_speaker = None

    for result in response.results:
        # Get the transcript and speaker tags
        alternative = result.alternatives[0]
        for word_info in alternative.words:
            speaker_tag = speaker_labels.get(word_info.speaker_tag, f"Speaker {word_info.speaker_tag}")

            # Check for a new speaker
            if speaker_tag != current_speaker:
                if current_sentence:
                    sentences.append({
                        'speaker': current_speaker,
                        'sentence': ' '.join(current_sentence),
                    })
                    current_sentence = []
                current_speaker = speaker_tag

            current_sentence.append(word_info.word)

            # End of sentence punctuation detection
            if word_info.word.endswith('.') or word_info.word.endswith('?') or word_info.word.endswith('!'):
                sentences.append({
                    'speaker': current_speaker,
                    'sentence': ' '.join(current_sentence),
                })
                current_sentence = []

    # Add the last sentence if there are any left
    if current_sentence:
        sentences.append({
            'speaker': current_speaker,
            'sentence': ' '.join(current_sentence),
        })

    # Prepare output
    output = []
    for sentence in sentences:
        speaker_label = sentence['speaker']
        text = sentence['sentence']
        output.append(f"{speaker_label}: {text}")

    return jsonify({'transcription': output})

if __name__ == '__main__':
    app.run(debug=True, port=8962)
