import keyboard
import queue
import threading
import argparse
import tempfile
import queue
import sys
import sounddevice as sd
import soundfile as sf
import numpy  

class SpacebarPress(Exception):
    pass

def listen_for_spacebar():
    try:
        while True:
            if keyboard.is_pressed('space'):  # if key 'space' is pressed 
                raise SpacebarPress
    except SpacebarPress:
        return  # Just exit the function, which will stop the thread


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', nargs='?', metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
parser.add_argument(
    '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)

q = queue.Queue()


def callback(indata, frames, time, status): #it gets inputs from the inputStream object
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = int(device_info['default_samplerate'])

    if args.filename is None:
        args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
                                        suffix='.wav', dir='')

    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate, #This line uses the soundfile.SoundFile class to open a sound file. It is opened in 'x' mode, which means that a new file will be created
                      channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,    #sounddevice.InputStream class to open an audio input stream.
                            channels=args.channels, callback=callback, blocksize=1024) as stream:
            print('#' * 80)
            print('press Spacebar to stop the recording')
            print('#' * 80)

            # Start the listener on a separate thread
            listener_thread = threading.Thread(target=listen_for_spacebar)
            listener_thread.start()

            while True:
                # No longer using get_nowait(), which should prevent the buzzing sound
                file.write(q.get())
                # Check if the listener thread has stopped
                if not listener_thread.is_alive():
                    break  # Stop the loop and finish processing the buffered audio data
                
            # Make sure all buffered audio data is processed before stopping the recording
            while not q.empty():
                file.write(q.get())
                
except SpacebarPress:
    print('Spacebar pressed, stopping recording...')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

print("test")