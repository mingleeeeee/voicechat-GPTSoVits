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
   git clone https://github.com/yourusername/tts-flask-server.git
   cd tts-flask-server
   ```
2. **Set Up Virtual Environment**
  ```bash
  pip install -r requirements.txt
  ```
3. **Install Dependencies**
   ```bash
   conda create -n GPTSoVits python=3.9
   conda activate GPTSoVits
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables**
  ```bash
  nano .env
  OPENAI_API_KEY=your_openai_api_key
  AWS_ACCESS_KEY_ID=ID
  AWS_SECRET_ACCESS_KEY=KEY
  AWS_REGION=REGION
  ```
5. **Download Models**
  - Extract under GPT-SoViTS Folder
  https://drive.google.com/file/d/1xEXZdxATBUj70MOM_RMcUy0V50i111hP/view?usp=sharing
6. **Run the Server**
  ```bash
  python app.py
  ```