# app.py
import nltk
import os
import soundfile as sf
import random
import uuid
import torch
from flask import Flask, render_template, request, jsonify
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
import math
import string
from langdetect import detect  # Detect the user's language

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

## Deleted statement: Tokyobeast related
# # Load system content from WHITEPAPER.txt
# system_content = "2文以内で、できる限り短く、日本語や中国語や英語で答えてください。詳細な説明は省略し、要点のみを述べてください。:"
# # Define RAG file
# with open("RAG/WHITEPAPER.txt", "r", encoding="utf-8") as f:
#     system_content += f.read()
##

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
    {"role": "system", "content": ""}  # Include the initial system message
]

# def get_chatbot_response(user_input, chat_history):
#     # Append the user's message to the chat history
#     chat_history.append({"role": "user", "content": user_input})
    
#     # Call the OpenAI API with the chat history
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=chat_history
#     )
    
#     # Extract the chatbot's response
#     bot_response = response.choices[0].message.content.strip()
    
#     # Append the bot's response to the chat history
#     chat_history.append({"role": "assistant", "content": bot_response})
    
#     # Return the response and updated chat history
#     return title_case_all_upper_phrases(bot_response), chat_history

def get_chatbot_response(user_input, chat_history):
    # Append the user's message to the chat history
    chat_history.append({"role": "user", "content": user_input})

    # Instruction for GPT to classify intent
    system_instruction = """
    You are an assistant designed to understand user queries about products.
    If the user wants to search for a product, respond only with: 
    "Searching for <keyword>". 
    Do not provide additional recommendations or examples.
    """
    chat_history.append({"role": "system", "content": system_instruction})

    # Call GPT for intent classification
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=chat_history
    )

    # Extract GPT's response
    bot_response = response.choices[0].message.content.strip()

    # Identify "Searching for" intent
    if "searching for" in bot_response.lower():
        product_keyword_match = re.search(r"searching for (.+)", bot_response, re.IGNORECASE)
        if product_keyword_match:
            # Extract and clean the product keyword
            product_keyword = product_keyword_match.group(1).strip().lower()
            product_keyword = product_keyword.strip(string.punctuation)  # Remove trailing punctuation

            # Search products in the backend
            matching_products = [
                {"id": pid, **details}
                for pid, details in product_list.items()
                if product_keyword in details["name"].lower()
            ]

            # Build response based on search results
            if matching_products:
                product_list_text = "\n".join(
                    [f"{product['name']} - ${product['price']} ({product['category']})" for product in matching_products]
                )
                shop_response = f"Here are the products I found for {product_keyword}:\n{product_list_text}"
            else:
                shop_response = f"Sorry, I couldn't find any products matching {product_keyword}"

            # Append and return the response
            chat_history.append({"role": "assistant", "content": shop_response})
            return shop_response, chat_history

    # Default fallback response for non-product queries
    chat_history.append({"role": "assistant", "content": bot_response})
    return bot_response, chat_history

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

###################Start: User Agent - Shop Agent #########################

# Product data
product_list = {
    1: {"name": "Smartphone X", "price": 699, "category": "Electronics"},
    2: {"name": "Gaming Laptop", "price": 1299, "category": "Computers"},
    3: {"name": "Wireless Earbuds", "price": 199, "category": "Accessories"},
    4: {"name": "Smartwatch Y", "price": 249, "category": "Wearables"},
    5: {"name": "4K TV", "price": 899, "category": "Home Appliances"},
    6: {"name": "Bluetooth Speaker", "price": 99, "category": "Accessories"},
    7: {"name": "DSLR Camera", "price": 999, "category": "Photography"},
    8: {"name": "Electric Kettle", "price": 49, "category": "Kitchen Appliances"},
    9: {"name": "Vacuum Cleaner", "price": 299, "category": "Home Appliances"},
    10: {"name": "Air Conditioner", "price": 1200, "category": "Home Appliances"},
    11: {"name": "Microwave Oven", "price": 200, "category": "Kitchen Appliances"},
    12: {"name": "Gaming Keyboard", "price": 150, "category": "Computers"},
    13: {"name": "Wireless Mouse", "price": 30, "category": "Computers"},
    14: {"name": "LED Monitor", "price": 250, "category": "Computers"},
    15: {"name": "Portable Charger", "price": 50, "category": "Accessories"},
    16: {"name": "Smart Door Lock", "price": 199, "category": "Home Security"},
    17: {"name": "Robot Vacuum", "price": 499, "category": "Home Appliances"},
    18: {"name": "Streaming Device", "price": 129, "category": "Electronics"},
    19: {"name": "Noise Cancelling Headphones", "price": 299, "category": "Accessories"},
    20: {"name": "Smart Thermostat", "price": 249, "category": "Home Security"},
    21: {"name": "Digital Watch", "price": 79, "category": "Wearables"},
    22: {"name": "E-Reader", "price": 129, "category": "Electronics"},
    23: {"name": "Desktop Computer", "price": 899, "category": "Computers"},
    24: {"name": "3D Printer", "price": 1200, "category": "Computers"},
    25: {"name": "Noise Machine", "price": 59, "category": "Home Appliances"},
    26: {"name": "Smart Light Bulb", "price": 19, "category": "Home Security"},
    27: {"name": "Electric Toothbrush", "price": 49, "category": "Personal Care"},
    28: {"name": "Fitness Tracker", "price": 99, "category": "Wearables"},
    29: {"name": "Gaming Headset", "price": 129, "category": "Computers"},
    30: {"name": "Electric Shaver", "price": 69, "category": "Personal Care"},
    31: {"name": "Cordless Drill", "price": 149, "category": "Tools"},
    32: {"name": "Home Security Camera", "price": 199, "category": "Home Security"},
    33: {"name": "Tablet Device", "price": 499, "category": "Electronics"},
    34: {"name": "Portable Projector", "price": 599, "category": "Electronics"},
    35: {"name": "Smartphone Y", "price": 799, "category": "Electronics"},
    36: {"name": "Action Camera", "price": 249, "category": "Photography"},
    37: {"name": "Compact Dishwasher", "price": 499, "category": "Kitchen Appliances"},
    38: {"name": "Air Fryer", "price": 199, "category": "Kitchen Appliances"},
    39: {"name": "Instant Pot", "price": 129, "category": "Kitchen Appliances"},
    40: {"name": "Electric Scooter", "price": 399, "category": "Transportation"},
    41: {"name": "Smartphone Z", "price": 899, "category": "Electronics"},
    42: {"name": "Laptop X", "price": 1499, "category": "Computers"},
    43: {"name": "Laptop Y", "price": 999, "category": "Computers"},
    44: {"name": "Standing Desk", "price": 399, "category": "Furniture"},
    45: {"name": "Ergonomic Chair", "price": 299, "category": "Furniture"},
    46: {"name": "Coffee Maker", "price": 99, "category": "Kitchen Appliances"},
    47: {"name": "Blender", "price": 89, "category": "Kitchen Appliances"},
    48: {"name": "Electric Grill", "price": 149, "category": "Kitchen Appliances"},
    49: {"name": "Juicer", "price": 99, "category": "Kitchen Appliances"},
    50: {"name": "Printer", "price": 149, "category": "Computers"},
}

# Helper function: Pagination
def paginate(data, page, per_page):
    total_items = len(data)
    total_pages = math.ceil(total_items / per_page)
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    return {
        "items": data[start_index:end_index],
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": page,
    }

# Endpoint: Search Products
@app.route('/user_agent/search', methods=['GET'])
def search_products():
    query = request.args.get('query', '').lower()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 2))

    # Filter products by query
    results = [
        {"id": pid, **details}
        for pid, details in product_list.items()
        if query in details['name'].lower()
    ]

    # Paginate results
    paginated = paginate(results, page, per_page)
    return jsonify(paginated)

# Endpoint: Get Structured Product Data
@app.route('/user_agent/product/<int:product_id>', methods=['GET'])
def get_product_data(product_id):
    product = product_list.get(product_id)
    if not product:
        return jsonify({"error": "Product not found"}), 404

    # Mock structured data
    structured_data = {
        "id": product_id,
        "details": product,
        "metadata": {
            "rating": round(random.uniform(3.0, 5.0), 1),
            "reviews": random.randint(100, 500),
        },
    }
    return jsonify(structured_data)

# Shop Agent Endpoint: Mock Additional Data
@app.route('/shop_agent/<int:product_id>', methods=['GET'])
def shop_agent(product_id):
    # Return mock data
    return jsonify({
        "product_id": product_id,
        "additional_info": f"Mock data for product {product_id} from Shop Agent.",
    })

# User Agent Endpoint: Combine Data with Shop Agent
@app.route('/user_agent/product_with_details/<int:product_id>', methods=['GET'])
def product_with_details(product_id):
    product = product_list.get(product_id)
    if not product:
        return jsonify({"error": "Product not found"}), 404

    # Call Shop Agent for more details
    shop_agent_response = {
        "additional_info": f"Mock data for product {product_id} from Shop Agent.",
    }

    combined_data = {
        "id": product_id,
        "details": product,
        "shop_agent_data": shop_agent_response["additional_info"],
    }
    return jsonify(combined_data)

###################End: User Agent - Shop Agent #########################

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)