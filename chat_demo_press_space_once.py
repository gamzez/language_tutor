import speech_recognition as sr
import whisper
import numpy as np
import tempfile
import os
import torch
import openai
import argparse
from gtts import gTTS
import pygame
import time
import pyaudio
import audioread
import pickle
import press2talk
import keyboard
import queue
import threading
import sys
import sounddevice as sd
import soundfile as sf
language = 'de'

#read the txt file contains OpenAI API key
with open('openai_api_key.txt.gitignore') as f:
    api_key = f.readline()
openai.api_key = api_key

recording = False  # The flag indicating whether we're currently recording
done_recording = False  # Flag indicating the user has finished recording

def listen_for_spacebar():
    global recording, done_recording
    space_press_count = 0
    while True:
        if keyboard.is_pressed('space'):  # if key 'space' is pressed
            time.sleep(0.2)  # Debounce to avoid multiple detections
            space_press_count += 1
            if space_press_count % 2 != 0:  # Odd number of presses
                print("Space bar pressed! Recording started.")
                recording = True
                done_recording = False
            else:  # Even number of presses
                print("Space bar pressed again! Recording stopped.")
                recording = False
                done_recording = True
                break  # Exit the thread

      
def get_sample_rate(file_name):
    with audioread.audio_open(file_name) as audio_file:
        return audio_file.samplerate

def read_wav_file(file_name):
    with audioread.audio_open(file_name) as audio_file:
        audio_data = b""
        for buf in audio_file:
            audio_data += buf
    return audio_data

def play_wav_once(file_name, speed=1.0):
    pygame.init()
    pygame.mixer.init()

    try:
        # Read the .wav file using audioread
        audio_data = read_wav_file(file_name)
        sample_rate = 26400 #get_sample_rate(file_name)
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



#end


def press2record(filename, subtype, channels, samplerate = 24000):

    q = queue.Queue()
    #func start
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())
    global recording, done_recording
    print(f"recording: {recording}")
    print(f"done_recording: {done_recording}")
    try:
        if samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            samplerate = int(device_info['default_samplerate'])
            print(int(device_info['default_samplerate']))
        # if filename is None:
        #     filename = tempfile.mktemp(prefix='captured_audio',
        #                                     suffix='.wav', dir='')

        with sf.SoundFile(filename, mode='x', samplerate=samplerate,
                        channels=channels, subtype=subtype) as file:
            with sd.InputStream(samplerate=samplerate, device=args.device,
                                channels=channels, callback=callback, blocksize=4096) as stream:
                print('#' * 80)
                print('press Spacebar to start recording, release to stop')
                print('#' * 80)
                import time 
                time.sleep(15)

                # Start the listener on a separate thread
                listener_thread = threading.Thread(target=listen_for_spacebar)
                listener_thread.start()

                while not done_recording:
                    while recording and not q.empty():
                        file.write(q.get())
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
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
    print(f"file saved to {saved_file}")
    # Transcribe the temporary WAV file using Whisper
    result = audio_model.transcribe(saved_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    # Delete the temporary WAV file
    os.remove(saved_file)
    print(f"text: {text}")
    return text
    


def ask_to_continue():
    while True:
        # Ask the user if they want to continue conversation
        continue_flag = input("Do you want to continue to edit? Please enter 'y' or 'n': ").strip()
        if continue_flag == "y":
            return True # Return True to indicate the user wants to continue editing
            break
        elif continue_flag == "n":
            return False # Return False to indicate the user does not want to continue editing
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def interact_with_tutor(timeout):
    # Define the system role to set the behavior of the chat assistant
    messages = [
        {"role": "system", "content" : "Du bist Anna, meine deutsche Lernpartnerin. Du wirst mit mir chatten, als wärst du Ende 20. Das Thema ist das Leben in Deutschland. Ihre Antworten werden kurz und einfach sein. Mein Niveau ist B1, stellen Sie Ihre Satzkomplexität auf mein Niveau ein. Versuche immer, mich zum Reden zu bringen, indem du Fragen stellst, und vertiefe den Chat immer."}
    ]
    while True:
        # Get the user's voice command
        command = get_voice_command()  
        print(command)
        # Add the user's command to the message history
        messages.append({"role": "user", "content": command})  
        # Generate a response from the chat assistant
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )  
        
        chat_response = completion.choices[0].message.content  # Extract the response from the completion
        print(f'ChatGPT: {chat_response}')  # Print the assistant's response
        messages.append({"role": "assistant", "content": chat_response})  # Add the assistant's response to the message history
        myobj = gTTS(text=messages[-1]['content'],tld="de", lang=language, slow=False)
        myobj.save("/Users/gamze/Desktop/language_tutor/welcome.wav")
        current_dir = os.getcwd()
        audio_file = '/Users/gamze/Desktop/language_tutor/welcome.wav'
        play_wav_once(audio_file, 1.0)
        os.remove(audio_file)
        continue_flag = ask_to_continue()  # Prompt the user to continue or not
        if not continue_flag:
            save_response_to_pkl(messages)
            save_response_to_txt(messages)
            return chat_response  # Return the final chat response and exit the loop
            break

            
if __name__ == "__main__":
    if os.path.exists("input_to_gpt.wav"):
        os.remove("input_to_gpt.wav")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--timeout", default=2, type=float, help="Timeout for stopping transcription")
    parser.add_argument('-d', '--device', type=int_or_str,help='input device (numeric ID or substring)')
    parser.add_argument('-r', '--samplerate', default=24000, type=int, help='sampling rate')
    parser.add_argument(
        '-c', '--channels', type=int, default=1, help='number of input channels')
    parser.add_argument(
        '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args()
    model = args.model # + ".en"
    audio_model = whisper.load_model(model)
    tutor_response = interact_with_tutor(args.timeout)
    print('GPT response:')
    print(f'{tutor_response}')






