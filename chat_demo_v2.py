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

#Read the txt file contains OpenAI API key
with open('openai_api_key.txt') as f:
    api_key = f.readline()
openai.api_key = api_key

recording = False       # Indicates whether the system is currently recording audio
done_recording = False  # Indicates that the user has completed recording a voice command 
stop_recording = False  # Indicates that the user wants to exit the conversation

def listen_for_keys():
    # Function to listen for key presses to control recording
    global recording, done_recording, stop_recording
    while True:
        if keyboard.is_pressed('space'):  # Start recording on spacebar press
            stop_recording = False
            recording = True
            done_recording = False
        elif keyboard.is_pressed('esc'):  # Stop recording on 'esc' press
            stop_recording = True
            break
        elif recording:  # Stop recording on spacebar release
            recording = False
            done_recording = True
            break
        time.sleep(0.01)


def get_sample_rate(file_name):
    # Function to get the sample rate of an audio file.
    with audioread.audio_open(file_name) as audio_file:
        return audio_file.samplerate

def read_wav_file(file_name):
    # Function to read a WAV file and return its audio data.
    with audioread.audio_open(file_name) as audio_file:
        audio_data = b""
        for buf in audio_file:
            audio_data += buf
    return audio_data

def callback(indata, frames, time, status):
    # Function called for each audio block during recording.
    if recording:  # Only record if the recording flag is set
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())  # Store the audio data

def play_wav_once(file_name, sample_rate, speed=1.0):
    # Function to play a WAV file once at a given sample rate and speed.
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
    # Function to handle recording when a key is pressed
    global recording, done_recording, stop_recording
    stop_recording = False
    recording = False
    done_recording = False
    try:
        # Determine the samplerate if not provided
        if samplerate is None:
            device_info = sd.query_devices(None, 'input')
            samplerate = int(device_info['default_samplerate'])
            print(int(device_info['default_samplerate']))
        # Create a temporary filename if not provided
        if filename is None:
            filename = tempfile.mktemp(prefix='captured_audio',
                                       suffix='.wav', dir='')
        # Open the sound file for writing
        with sf.SoundFile(filename, mode='x', samplerate=samplerate,
                          channels=channels, subtype=subtype) as file:
            with sd.InputStream(samplerate=samplerate, device=None,
                                channels=channels, callback=callback, blocksize=4096) as stream:
                print('press Spacebar to start recording, release to stop, or press Esc to exit')
                listener_thread = threading.Thread(target=listen_for_keys)  # Start the listener on a separate thread
                listener_thread.start()
                # Write the recorded audio to the file
                while not done_recording and not stop_recording:
                    while recording and not q.empty():
                        file.write(q.get())
        # Return -1 if recording is stopped
        if stop_recording:
            return -1

    except KeyboardInterrupt:
        print('Interrupted by user')

    return filename


def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text
        
def get_voice_command():
    # Function to capture and transcribe the user's voice command.
    global done_recording, recording
    done_recording = False
    recording = False

    # Call the press2record function to capture the user's voice command
    saved_file = press2record(filename="input_to_gpt.wav", subtype=args.subtype, channels=args.channels, samplerate=args.samplerate)
    
    # return -1 if recording was stopped
    if saved_file == -1:
        return -1

    # Transcribe the temporary WAV file using Whisper
    result = audio_model.transcribe(saved_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()

    # Delete the temporary WAV file
    os.remove(saved_file)

    # Print the transcribed text
    print(f"\nYou: {text} \n")

    return text  # Return the transcribed text



def interact_with_tutor():
    # Define the system role to set the behavior of the chat assistant
    messages = [
        {"role": "system", "content": "Du bist Anna, meine deutsche Lernpartnerin. Du wirst mit mir chatten, als wärst du eine Freundin von mir. Ich werde dir sagen an welchem Thema ich reden wollte. Ihre Antworten werden kurz und einfach sein. Mein Niveau ist B1, stell deine Satzkomplexität auf mein Niveau ein. Versuche immer, mich zum Reden zu bringen, indem du Fragen stellst, und vertiefe den Chat immer."}
    ]

    while True:
        # Get the user's voice command
        command = get_voice_command()
        if command == -1:
            # Save the chat logs and exit if recording is stopped
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

        # Extract the response from the completion
        chat_response = completion.choices[0].message.content
        print(f'ChatGPT: {chat_response} \n')  # Print the assistant's response

        #Append the response to the messages with the role "assistant" to store the chat history.
        messages.append({"role": "assistant", "content": chat_response})

        # Convert the text response to speech
        speech_object = gTTS(text=messages[-1]['content'], tld="de", lang=language, slow=False)
        speech_object.save("/Users/gamze/Desktop/language_tutor/welcome.wav")
        current_dir = os.getcwd()
        audio_file = '/Users/gamze/Desktop/language_tutor/welcome.wav'

        # Play the audio response
        play_wav_once(audio_file, args.samplerate, 1.0)
        os.remove(audio_file)  # Remove the temporary audio file

         
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

