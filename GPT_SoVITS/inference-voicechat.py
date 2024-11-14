# inference-voicechat.py
import os
import sys
import soundfile as sf
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

i18n = I18nAuto()

def synthesize(text, output_path):
    # Ensure output_path is not empty and includes a filename
    if not output_path or os.path.isdir(output_path):
        raise ValueError("Error: `output_path` must be a full path, including a filename.")

    # Paths for model files
    GPT_model_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert.ckpt"
    SoVITS_model_path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    ref_audio_path = "GPT_SoVITS/inference/train.wav_0000112640_0000241920.wav"
    ref_text_path = "GPT_SoVITS/inference/ref_text.txt"
    ref_language = "日文"
    target_language = "日文"

    # Load reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Generate audio based on provided text
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=text,
        text_language=i18n(target_language),
        top_p=1,
        temperature=1
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the audio file to the specified output path
        sf.write(output_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_path}")
        return output_path
    else:
        print("Error: Audio generation failed.")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python inference-voicechat.py '<text>' '<output_path>'")
        sys.exit(1)

    input_text = sys.argv[1]
    output_path = sys.argv[2]

    # Run synthesis with specified input and output path
    synthesize(input_text, output_path)
