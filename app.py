import os
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

load_dotenv()

generation_config = {
    "max_output_tokens": 50,  # Adjust as needed for desired output length
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

vertexai.init(project=os.getenv('GCP_PROJECT_NAME'), location="us-central1")
model = GenerativeModel("gemini-1.5-flash-preview-0514")

app = Flask(__name__)
CORS(app)  # Enables CORS for cross-origin requests from your front-end

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text_to_translate = data.get('text')
    target_language = data.get('language')
    if(target_language == 'ta'):
        target_language = 'ta-IN'
    if not text_to_translate:
        return jsonify({'error': 'No text provided'}), 400
    if not target_language:
        return jsonify({'error': 'No target language provided'}), 400
    prompt = f"Translate to {target_language}: {text_to_translate}"
    translated_text = generate_translation(prompt)
    return jsonify({'translated_text': translated_text})

def generate_translation(prompt):
    responses = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    translated_text = ""
    for response in responses:
        translated_text += response.text
    return translated_text

if __name__ == '__main__':
    app.run(debug=True, port=5000)
