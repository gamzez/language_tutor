## ask_to_continue ve press2recordun ikisi de var.
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import speech_recognition as sr
import whisper
import numpy as np
import tempfile
import torch
import openai
import argparse
from gtts import gTTS
import pygame
import time
import pyaudio
import audioread
import pickle
import keyboard
import queue
import threading
import sys
import sounddevice as sd
import soundfile as sf
language = 'de'

#read the txt file contains OpenAI API key
with open('openai_api_key.txt') as f:
    api_key = f.readline()
openai.api_key = api_key

recording = False  # The flag indicating whether we're currently recording
done_recording = False  # Flag indicating the user has finished recording
stop_recording = False

def listen_for_keys():
    global recording, done_recording, stop_recording
    while True:
        if keyboard.is_pressed('space'):  # if key 'space' is pressed
            stop_recording = False
            recording = True
            done_recording = False
        elif keyboard.is_pressed('esc'):  # if key 'esc' is pressed
            stop_recording = True
            break  # Exit the thread
        elif recording:  # if key 'space' was released after recording
            recording = False
            done_recording = True
            break  # Exit the thread
        time.sleep(0.01)

def get_sample_rate(file_name):
    with audioread.audio_open(file_name) as audio_file:
        return audio_file.samplerate

def read_wav_file(file_name):
    with audioread.audio_open(file_name) as audio_file:
        audio_data = b""
        for buf in audio_file:
            audio_data += buf
    return audio_data

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if recording:  # Only record if the recording flag is set
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

def play_wav_once(file_name, sample_rate, speed=1.0):
    pygame.init()
    pygame.mixer.init()

    try:
        # Read the .wav file using audioread
        audio_data = read_wav_file(file_name)
        #sample_rate = 26400 #get_sample_rate(file_name)
        # Convert the raw audio data to a NumPy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Adjust the playback speed using NumPy's resample function
        adjusted_audio = np.interp(
            np.linspace(0, len(audio_array), int(len(audio_array) / speed)),
            np.arange(len(audio_array)),
            audio_array
        ).astype(np.int16)

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=int(sample_rate * speed),
                        output=True)

        # Play the sound
        stream.write(adjusted_audio.tobytes())

        # Close the audio stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    except pygame.error as e:
        print(f"Error playing the .wav file: {e}")
    finally:
        pygame.mixer.quit()

def save_response_to_pkl(chat):
    with open("chat_logs/chat_log2.pkl", 'wb') as file:
        pickle.dump(chat, file)


def save_response_to_txt(chat):        
    with open("chat_logs/chat_log2.txt", "w", encoding="utf-8") as file:
        for chat_entry in chat:
            role = chat_entry["role"]
            content = chat_entry["content"]
            file.write(f"{role}: {content}\n")


def press2record(filename, subtype, channels, samplerate=24000):
    global recording, done_recording, stop_recording
    stop_recording = False
    recording = False
    done_recording = False
    try:
        if samplerate is None:
            device_info = sd.query_devices(None, 'input')
            samplerate = int(device_info['default_samplerate'])
            print(int(device_info['default_samplerate']))
        if filename is None:
            filename = tempfile.mktemp(prefix='captured_audio',
                                       suffix='.wav', dir='')

        with sf.SoundFile(filename, mode='x', samplerate=samplerate,
                          channels=channels, subtype=subtype) as file:
            with sd.InputStream(samplerate=samplerate, device=None,
                                channels=channels, callback=callback, blocksize=4096) as stream:
                print('press Spacebar to start recording, release to stop, or press Esc to exit')
                listener_thread = threading.Thread(target=listen_for_keys)  # Start the listener on a separate thread
                listener_thread.start()
                while not done_recording and not stop_recording:
                    while recording and not q.empty():
                        file.write(q.get())

        if stop_recording:
            return -1

    except KeyboardInterrupt:
        print('Interrupted by user')

    return filename


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text
        
def get_voice_command():
    global done_recording, recording
    done_recording = False
    recording = False

    saved_file = press2record(filename="input_to_gpt.wav", subtype = args.subtype, channels = args.channels, samplerate = args.samplerate)
    
    if saved_file == -1:
        return -1
    # Transcribe the temporary WAV file using Whisper
    result = audio_model.transcribe(saved_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    # Delete the temporary WAV file
    os.remove(saved_file)
    print(f"\nYou: {text} \n")
    return text
    

def interact_with_tutor():
    # Define the system role to set the behavior of the chat assistant
    messages = [
        {"role": "system", "content" : "Du bist Anna, meine deutsche Lernpartnerin. Du wirst mit mir chatten, als wärst du Ende 20. Das Thema ist das Leben in Deutschland. Ihre Antworten werden kurz und einfach sein. Mein Niveau ist B1, stellen Sie Ihre Satzkomplexität auf mein Niveau ein. Versuche immer, mich zum Reden zu bringen, indem du Fragen stellst, und vertiefe den Chat immer."}
    ]
    while True:

        # Get the user's voice command
        command = get_voice_command()  
        if command == -1:
            save_response_to_pkl(messages)
            save_response_to_txt(messages)
            return "Chat has been stopped."
        # Add the user's command to the message history
        messages.append({"role": "user", "content": command})  
        # Generate a response from the chat assistant
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )  
        
        chat_response = completion.choices[0].message.content  # Extract the response from the completion
        print(f'ChatGPT: {chat_response} \n')  # Print the assistant's response
        messages.append({"role": "assistant", "content": chat_response})  # Add the assistant's response to the message history
        myobj = gTTS(text=messages[-1]['content'],tld="de", lang=language, slow=False)
        myobj.save("/Users/gamze/Desktop/language_tutor/welcome.wav")
        current_dir = os.getcwd()
        audio_file = '/Users/gamze/Desktop/language_tutor/welcome.wav'
        play_wav_once(audio_file, args.samplerate, 1.0)
        os.remove(audio_file)
        
            
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
    tutor_response = interact_with_tutor()
    print ("\033[A                             \033[A")

