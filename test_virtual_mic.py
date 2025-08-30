import sounddevice as sd
import numpy as np
import time
import soundfile as sf

# --- Configuration ---
# Windows: 'CABLE Input (VB-Audio Virtual Cable)'
VIRTUAL_MIC_NAME = 'CABLE In'
# macOS: 'BlackHole 2ch'
# Linux: 'my_virtual_mic' or the name you chose

# Path to the audio file
AUDIO_FILE_PATH = r"identify_sample\ta.wav"
SAMPLE_RATE = 48000  # Samples per second (Hz)
BLOCK_SIZE = 1024  # Number of samples per block


def find_device_id(device_name, kind):
    """Finds the ID of an audio device by its name."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        # A bit of fuzzy matching
        if device_name.lower() in device['name'].lower() and device[f'max_{kind}_channels'] > 0:
            print(f"Found device '{device['name']}' with ID {i}.")
            return i
    return None


def main():
    """
    Reads audio from a file and streams it to the virtual audio device.
    """
    # 1. Load the audio file
    try:
        print(f"Loading audio file: {AUDIO_FILE_PATH}")
        audio_data, file_sample_rate = sf.read(AUDIO_FILE_PATH)
        print(f"Audio file loaded successfully. Original sample rate: {file_sample_rate} Hz")
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("Converted stereo to mono")
        
        # Resample if necessary
        if file_sample_rate != SAMPLE_RATE:
            print(f"Resampling from {file_sample_rate} Hz to {SAMPLE_RATE} Hz")
            # Simple resampling (for better quality, consider using scipy.signal.resample)
            ratio = SAMPLE_RATE / file_sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(np.linspace(0, len(audio_data), new_length), 
                                 np.arange(len(audio_data)), audio_data)
        
        print(f"Audio duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Make sure the file exists and is a supported audio format.")
        return

    # 2. Find the virtual microphone's output device (the "sink")
    device_id = find_device_id(VIRTUAL_MIC_NAME, 'output')

    if device_id is None:
        print(
            f"\nError: Could not find the virtual audio device named '{VIRTUAL_MIC_NAME}'.")
        print("Please ensure the driver is installed and check the name.")
        print("Available devices are:")
        print(sd.query_devices())
        return

    # 3. Setup the audio stream
    current_frame = 0

    def audio_callback(outdata, frames, time, status):
        """This function is called by the sounddevice stream to get more audio data."""
        nonlocal current_frame
        if status:
            print(status)

        # Calculate the end frame for this block
        end_frame = current_frame + frames
        
        # Check if we have enough audio data left
        if current_frame >= len(audio_data):
            # If we've reached the end, loop back to the beginning
            current_frame = 0
            end_frame = frames
        
        # Handle case where we need to wrap around
        if end_frame <= len(audio_data):
            # Simple case: just copy the data
            chunk = audio_data[current_frame:end_frame]
        else:
            # We need to wrap around to the beginning
            remaining = len(audio_data) - current_frame
            chunk1 = audio_data[current_frame:]
            chunk2 = audio_data[:frames - remaining]
            chunk = np.concatenate([chunk1, chunk2])
            current_frame = frames - remaining - 1  # Will be incremented below
        
        # Reshape for output (add channel dimension)
        outdata[:len(chunk), 0] = chunk
        
        # Pad with zeros if chunk is shorter than requested frames
        if len(chunk) < frames:
            outdata[len(chunk):, 0] = 0
        
        # Update the current frame for the next block
        current_frame = (current_frame + frames) % len(audio_data)

    # 4. Start streaming
    try:
        print(f"\nStreaming audio from '{AUDIO_FILE_PATH}' to '{VIRTUAL_MIC_NAME}'...")
        print("The audio will loop continuously. Press Ctrl+C to stop.")

        # Create and start the stream
        with sd.OutputStream(device=device_id,
                             channels=1,
                             samplerate=SAMPLE_RATE,
                             callback=audio_callback):
            # The stream runs in the background. We just need to keep the script alive.
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
