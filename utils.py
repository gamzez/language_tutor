## ask_to_continue ve press2recordun ikisi de var.
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import tempfile
import torch
import openai
from gtts import gTTS
import pygame
import time
import pyaudio
import audioread
import pickle
import keyboard
import threading
import sys
import sounddevice as sd
import soundfile as sf

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

def callback(indata, frames, time, status, q):
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
    with open("chat_logs/chat_log.pkl", 'wb') as file:
        pickle.dump(chat, file)


def save_response_to_txt(chat):        
    with open("chat_logs/chat_log.txt", "w", encoding="utf-8") as file:
        for chat_entry in chat:
            role = chat_entry["role"]
            content = chat_entry["content"]
            file.write(f"{role}: {content}\n")


def press2record(q, filename, subtype, channels, samplerate=24000):

    def callback_with_q(indata, frames, time, status):
        return callback(indata, frames, time, status, q)
    
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
                                channels=channels, callback=callback_with_q, blocksize=4096) as stream:
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
        
def get_voice_command(args, q, audio_model):
    global done_recording, recording
    done_recording = False
    recording = False

    saved_file = press2record(q, filename="input_to_gpt.wav", subtype = args.subtype, channels = args.channels, samplerate = args.samplerate)
    
    if saved_file == -1:
        return -1
    # Transcribe the temporary WAV file using Whisper
    result = audio_model.transcribe(saved_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    # Delete the temporary WAV file
    os.remove(saved_file)
    #print ("\033[A                             \033[A")
    print(f"\nYou: {text} \n")
    return text
    

def interact_with_tutor(args, q, language, audio_model):
    # Define the system role to set the behavior of the chat assistant
    messages = [
        {"role": "system", "content" : "Du bist Anna, meine deutsch Tutor. Du wirst mit mir chatten, als wärst du eine Freundin von mir. Ich werde dir sagen an welchem Thema ich reden wollte. Ihre Antworten werden kurz (circa 30-40 Wörter) und einfach sein. Mein Niveau ist B1, stell deine Satzkomplexität auf mein Niveau ein. Versuche immer, mich zum Reden zu bringen, indem du Fragen stellst, und vertiefe den Chat immer.Du beantwortest nur auf Deutsch"}
    ]

    while True:

        # Get the user's voice command
        command = get_voice_command(args, q, audio_model)  
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
        speech_object = gTTS(text=messages[-1]['content'],tld="de", lang=language, slow=False)
        temporary_save_dir = "gtts.wav"
        speech_object.save(temporary_save_dir)
        play_wav_once(temporary_save_dir, args.samplerate, 1.0)
        os.remove(temporary_save_dir)
      