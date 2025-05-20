Here is a `README.md` file that provides setup instructions for your project:

```markdown
# Audio Transcription and Speaker Diarization API

This project is a Flask-based API that processes audio files, performs speaker diarization, transcribes audio using OpenAI's Whisper model, detects the language of the transcription, and translates it into English if necessary.

## Features

- Audio file handling from URL or direct upload.
- Speaker diarization using `pyannote.audio`.
- Transcription of audio files using OpenAI's Whisper API.
- Language detection using FastText.
- Spell checking using the `SpellChecker` library.
- Translation to English using OpenAI GPT-4.

## Requirements

- Python 3.8+
- OpenAI API Key for transcription and translation.
- Pretrained FastText language detection model (`lid.176.bin`).
- `pyannote.audio` pipeline for speaker diarization.
- A `.env` file to store your environment variables.

## Installation

1. **Upload and extract the zip file**

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create ENV FILE**
    
   Create a `.env` file in the root directory with the following environment variables:
   
5. **Set up environment variables**


   ```bash
   TRANSCRIPTION_API=<Your_OpenAI_API_Key>
   API_TOKEN=<Your_Custom_API_Token>
   USER_AUTH_TOKEN=<Your_Pyannote_Auth_Token>
   ```

6. **Set up directories**

   Create the following directory for temporary files:

   ```bash
   mkdir tmp
   ```

## Running the Application

To start the Flask API:

```bash
python app.py
```

The API will run on `http://0.0.0.0:8962` by default.

## API Endpoints

### 1. Spell Check

- **URL:** `/spellcheck`
- **Method:** `POST`
- **Description:** Checks the spelling of words in the provided text.

**Request Example:**

```json
{
  "text": "Ths is a smple text with erors."
}
```

**Response Example:**

```json
{
  "misspelled": [
    { "word": "Ths" },
    { "word": "smple" },
    { "word": "erors" }
  ]
}
```

### 2. Transcribe Audio

- **URL:** `/transcribe`
- **Method:** `POST`
- **Description:** Transcribes the provided audio file or audio URL, performs speaker diarization, detects language, and translates the transcription if necessary.

**Headers:**

```json
Authorization: <API_TOKEN>
```

**Request Example (file upload):**

```bash
curl -X POST -H "Authorization: <API_TOKEN>" -F "file=@path/to/file.wav" http://localhost:8962/transcribe
```

**Request Example (audio URL):**

```bash
curl -X POST -H "Authorization: <API_TOKEN>" -F "url=https://example.com/audio.wav" http://localhost:8962/transcribe
```

**Response Example:**

```json
{
  "converted_transcription": "00:10 Caller 1: Hello, how can I help you?",
  "original_language": "English (en)",
  "original_transcription": "00:10 Caller 1: Hello, how can I help you?",
  "converted_language": "English (en)",
  "transcription": "00:10 Caller 1: Hello, how can I help you?"
}
```

## Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── Sample/
│   └── lid.176.bin     # FastText language model
├── tmp/                # Directory for temporary files (audio, json)
├── README.md           # This file
├── .env                # Environment variables
```

## Dependencies

- Flask
- Flask-CORS
- Spacy (`en_core_web_sm`)
- Requests
- PyDub
- OpenAI Python client
- Pyannote Audio
- FastText
- Python-dotenv
- Pyspellchecker

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
```

### Notes:
1. Replace `<repository_url>` with your actual repository URL.
2. Add any additional dependencies to `requirements.txt` if not listed yet.