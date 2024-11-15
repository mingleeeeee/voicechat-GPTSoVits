# app.py
import nltk
import os
import soundfile as sf
import random
import uuid
import torch
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from openai import OpenAI
from tools.i18n.i18n import I18nAuto
from dotenv import load_dotenv
from GPT_SoVITS.inference_webui_fast import dict_language  # Ensure dict_language is imported
from GPT_SoVITS.inference_webui_fast import tts_pipeline, dict_language, version  # Ensure version is imported
import whisper
import io
import re
# Load Whisper model (choose model size based on performance needs)
model = whisper.load_model("small")  # Options: tiny, base, small, medium, large

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load system content from WHITEPAPER.txt
system_content = "2文以内で、できる限り短く、日本語や中国語や英語で答えてください。詳細な説明は省略し、要点のみを述べてください。:"
# Define RAG file
with open("RAG/WHITEPAPER.txt", "r", encoding="utf-8") as f:
    system_content += f.read()

# All uppercase will cause TTS model speak separately, like TOKYO would speak like T-O-K-Y-O, so transform TOKYO -> Tokyo to fix the problem.
def title_case_all_upper_phrases(text):
    # Split text by spaces but retain delimiters like punctuation
    parts = re.split(r'(\W+)', text)  # Split on non-word characters but keep them

    transformed_parts = []
    for part in parts:
        # Only transform if part is all-uppercase and doesn't contain non-Latin characters
        if part.isupper() and all(char.isalpha() or char.isspace() for char in part):
            transformed_parts.append(part.title())
        else:
            transformed_parts.append(part)

    return ''.join(transformed_parts)

# Initialize chat history as a global variable or session variable
chat_history = [
    {"role": "system", "content": system_content}  # Include the initial system message
]

def get_chatbot_response(user_input, chat_history):
    # Append the user's message to the chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Call the OpenAI API with the chat history
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history
    )
    
    # Extract the chatbot's response
    bot_response = response.choices[0].message.content.strip()
    
    # Append the bot's response to the chat history
    chat_history.append({"role": "assistant", "content": bot_response})
    
    # Return the response and updated chat history
    return title_case_all_upper_phrases(bot_response), chat_history


# Initialize TTS Pipeline with Absolute Paths
device = "cuda" if torch.cuda.is_available() else "cpu"
i18n = I18nAuto()

# Define the synthesize function for TTS synthesis (Text-To-Speech)
def synthesize_tts(text):
    # Set target and reference language based on available options
    target_language = "多語種混合" #On ec2 pls define as "Multilingual Mixed"
    ref_language = "日文" #On ec2 pls define as "Japanese"

    # Load reference text
    ref_text_path = "GPT_SoVITS/inference/ref_text.txt"                          #ref is Japanese Text
    ref_audio_path = "GPT_SoVITS/inference/train.wav_0000112640_0000241920.wav"  #ref is Japanese voice
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Check if target_language is valid
    if target_language not in dict_language:
        raise ValueError(f"Target language '{target_language}' is not supported by the TTS model.")

    # Configure input for TTS pipeline
    inputs = {
        "text": text,
        "text_lang": dict_language[target_language],
        "ref_audio_path": ref_audio_path,
        "prompt_text": ref_text,
        "prompt_lang": dict_language[ref_language],
        "top_p": 1,
        "temperature": 1,
    }

    # Run TTS synthesis and handle the result
    synthesis_result = list(tts_pipeline.run(inputs))
    if synthesis_result:
        sampling_rate, audio_data = synthesis_result[-1]
        return sampling_rate, audio_data
    else:
        print("Error: Audio generation failed.")
        return None, None

# Print statements to check values
print("Version:", version)
print("Available languages in dict_language:", dict_language.keys())
print("i18n('日文'):", i18n("日文"))
        
# Function to convert text to speech and save audio file
def text_to_speech_voicechat(text):
    unique_filename = f"output_{uuid.uuid4().hex}.wav"
    output_path = os.path.join("wav", unique_filename)

    # Run TTS synthesis
    sampling_rate, audio_data = synthesize_tts(text)

    if audio_data is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio_data, sampling_rate)
        print(f"Audio saved to {output_path}")
        return output_path
    else:
        print("Error: Failed to generate audio.")
        return None

# Function to convert STT (speech to text) through recording audio
@socketio.on("speech_to_text")
def handle_speech_to_text(data):
    audio_data = data.get("audio")
    if not audio_data:
        emit("stt_response", {"error": "No audio data received"})
        return

    # Process audio data
    audio_bytes = bytes(audio_data)
    audio_stream = io.BytesIO(audio_bytes)
    audio_file_path = "/tmp/audio.wav"
    
    # Save audio file for Whisper processing
    with open(audio_file_path, "wb") as f:
        f.write(audio_stream.read())

    # Perform STT with Whisper
    result = model.transcribe(audio_file_path)
    transcribed_text = result["text"]
    print("User speaks: "+transcribed_text)

    # Send the transcription back to the client
    if transcribed_text:
        emit("stt_response", {"text": transcribed_text})
    else:
        emit("stt_response", {"error": "Failed to transcribe audio"})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(msg):
    global chat_history
    # Get chatbot response
    bot_message, chat_history = get_chatbot_response(msg, chat_history)

    # Convert response to speech
    audio_path = text_to_speech_voicechat(bot_message)

    if audio_path:
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()

        # Emit both the message and audio data to the frontend
        emit('response_with_audio', {'message': bot_message, 'audio': audio_data})

        # Remove the audio file after sending it
        #os.remove(audio_path)
        #print(f"Temporary file {audio_path} removed.")
    else:
        emit('response_with_audio', {'message': bot_message, 'error': 'Failed to generate audio.'})

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)