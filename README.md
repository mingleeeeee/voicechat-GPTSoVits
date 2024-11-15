# TTS Flask Server

This project sets up a Flask server that provides a text-to-speech (TTS) interfac.
This project uses [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for text-to-speech synthesis. 

## Features

- SoVITS TTS synthesis for high-quality speech.
- Real-time response streaming.
- Frontend voicechat interaction with STT(Speech To Text).

## Prerequisites

- Python 3.8 or higher
- CUDA (if available) for GPU acceleration with PyTorch / CPU
- OpenAI API Key for text responses
- Pretrained SoVITS models for TTS synthesis

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mingleeeeee/voicechat-GPTSoVits.git
   cd VOICECHAT-GPTSOVITS
   ```
2. **Install Dependencies**
   ```bash
   conda create -n chatbot python=3.9
   conda activate chatbot
   pip install -r requirements.txt
   python download-nltk.py
   conda install ffmpeg
   ```
3. **Set Up Environment Variables**
  ```bash
  nano .env
  OPENAI_API_KEY=your_openai_api_key
  AWS_ACCESS_KEY_ID=ID
  AWS_SECRET_ACCESS_KEY=KEY
  AWS_REGION=REGION
  ```
5. **Download Models**
  - Extract 3 files into GPT-SoViTS Folder
  https://drive.google.com/file/d/1xEXZdxATBUj70MOM_RMcUy0V50i111hP/view?usp=sharing
6. **Run the Server**
  ```bash
  python app.py
  ```
7. **Note**
  1. Modify `system_content` in app.py to define chatbot behavior.
  2. Change `RAG file` to define extra information for chatbot.
  3. Change `target_language` under `synthesize_tts` function correspond to your computer language. (Can refer to log while server running)
  4. 