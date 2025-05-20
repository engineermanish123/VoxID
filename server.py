from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
from langdetect import detect

app = Flask(__name__)

LANGUAGE_MAP = {
    "en": "English", "fr": "French", "hi": "Hindi", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh-cn": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic", "tr": "Turkish", "nl": "Dutch", "el": "Greek",
    "sv": "Swedish", "pl": "Polish", "iw": "Hebrew", "bn": "Bengali", "th": "Thai",
    "id": "Indonesian", "vi": "Vietnamese", "ro": "Romanian", "fa": "Persian", "uk": "Ukrainian",
    "ur": "Urdu", "ta": "Tamil", "te": "Telugu", "ms": "Malay", "hu": "Hungarian"
}

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
    target_language = 'en'  # Always translate to English

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

if __name__ == '__main__':
    app.run(debug=True)