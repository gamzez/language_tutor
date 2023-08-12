import keyboard
import queue
import threading
import argparse
import tempfile
import sys
import sounddevice as sd
import soundfile as sf
import numpy
import time


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

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if recording:  # Only record if the recording flag is set
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())


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

if __name__ == "__main__":
    import os
    if os.path.exists("input_to_gpt.wav"):
        os.remove("input_to_gpt.wav")
    q = queue.Queue()
    saved_file = press2record(filename="input_to_gpt.wav", subtype=None,channels = 1, samplerate = 24000)
    print("saved_file: ", saved_file)