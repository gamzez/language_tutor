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
language = 'de'

#read the txt file contains OpenAI API key
with open('openai_api_key.txt.gitignore') as f:
    api_key = f.readline()
openai.api_key = api_key

      
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


        
def get_voice_command(timeout):
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        # adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        print("starts listening")
        try:
            # listen for speech for up to `timeout` seconds
            audio = r.listen(source, timeout)
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            return

        # Save audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio.get_wav_data())
        # Transcribe the temporary WAV file using Whisper
        result = audio_model.transcribe(temp_wav.name, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        # Delete the temporary WAV file
        os.remove(temp_wav.name)

        return text
    


def ask_to_continue():
    while True:
        # Ask the user if they want to continue conversation
        continue_flag = input("Do you want to continue to edit? Please enter 'y' or 'n': ")
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
        command = get_voice_command(timeout)  
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--timeout", default=2, type=float, help="Timeout for stopping transcription")
    args = parser.parse_args()
    model = args.model # + ".en"
    audio_model = whisper.load_model(model)
    tutor_response = interact_with_tutor(args.timeout)
    print('GPT response:')
    print(f'{tutor_response}')










