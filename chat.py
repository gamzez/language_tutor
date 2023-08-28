from utils import *
import speech_recognition as sr
import whisper
import argparse
import queue

language = 'de'

#read the txt file contains OpenAI API key
with open('openai_api_key.txt') as f:
    api_key = f.readline()
openai.api_key = api_key

# recording = False  # The flag indicating whether we're currently recording
# done_recording = False  # Flag indicating the user has finished recording
# stop_recording = False
            
if __name__ == "__main__":
    if os.path.exists("input_to_gpt.wav"):
        os.remove("input_to_gpt.wav")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument('-d', '--device', type=int_or_str,help='input device (numeric ID or substring)')
    parser.add_argument('-r', '--samplerate', default=27000, type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=1, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args()
    model = args.model 
    audio_model = whisper.load_model(model)
    q = queue.Queue()
    tutor_response = interact_with_tutor(args, q, language, audio_model)
    print ("\033[A                             \033[A")
