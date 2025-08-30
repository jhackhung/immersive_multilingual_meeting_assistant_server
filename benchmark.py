# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from datetime import datetime
import threading
import time
import tempfile
import os

import sounddevice as sd
from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.utils.onnx_torch_wrapper import (
    OnnxModelTorchWrapper,
    OnnxSessionOptions,
)

# Global variables for benchmark state
app = None
running = False
thread = None
transcription_count = 0

# Configuration
encoder_path = "build\\whisper_large_v3_turbo\\HfWhisperEncoder\\model.onnx"
decoder_path = "build\\whisper_large_v3_turbo\\HfWhisperDecoder\\model.onnx"
model_size = "large-v3-turbo"
audio_file = "D:\\segment.wav"

# Disable compile caching becuase Stable Diffusion is Pre-Compiled
# This is needed due to a bug in onnxruntime 1.22, and can be removed after the next ORT release.
options = OnnxSessionOptions.aihub_defaults()
options.context_enable = False

def load_model():
    """Load the Whisper model"""
    global app
    print("Loading model...")
    app = HfWhisperApp(
        OnnxModelTorchWrapper.OnNPU(encoder_path),
        OnnxModelTorchWrapper.OnNPU(decoder_path),
        f"openai/whisper-{model_size}",
    )
    print("Model loaded successfully!")

def transcription_loop():
    """Main transcription loop that runs in a separate thread"""
    global running, transcription_count, app
    
    while running:
        try:
            # Perform transcription
            print(f"Transcription #{transcription_count + 1} - Before: {datetime.now().astimezone()}")
            transcription = app.transcribe(audio_file)
            transcription_count += 1
            print(f"Transcription #{transcription_count}: {transcription}")
            print(f"After: {datetime.now().astimezone()}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            time.sleep(1)  # Wait a bit before retrying

def start_benchmark():
    """Start the transcription benchmark"""
    global running, thread, transcription_count, app
    
    if running:
        print("Benchmark is already running!")
        return
        
    if app is None:
        load_model()
        
    print("Starting transcription benchmark...")
    running = True
    transcription_count = 0
    thread = threading.Thread(target=transcription_loop, daemon=True)
    thread.start()
    print("Benchmark started! NPU utilization should now be high.")

def stop_benchmark():
    """Stop the transcription benchmark"""
    global running, thread, transcription_count
    
    if not running:
        print("Benchmark is not running!")
        return
        
    print("Stopping transcription benchmark...")
    running = False
    
    if thread and thread.is_alive():
        thread.join(timeout=5)
        
    print(f"Benchmark stopped. Total transcriptions completed: {transcription_count}")

def is_running():
    """Check if the benchmark is currently running"""
    global running
    return running

def get_transcription_count():
    """Get the current transcription count"""
    global transcription_count
    return transcription_count

def transcribe_wav(wav_input):
    """
    Transcribe a WAV file or WAV bytes data
    
    Args:
        wav_input (str or bytes): Path to the WAV file or WAV bytes data
        
    Returns:
        str: Transcription text or None if error
    """
    global app
    
    if app is None:
        print("Loading model for transcription...")
        load_model()
    
    try:
        if isinstance(wav_input, bytes):
            print(f"Transcribing WAV bytes data ({len(wav_input)} bytes)")
        else:
            print(f"Transcribing file: {wav_input}")
            
        start_time = datetime.now()
        transcription = app.transcribe(wav_input)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Transcription completed in {duration:.2f} seconds")
        print(f"Result: {transcription}")
        return transcription
        
    except Exception as e:
        input_type = "bytes data" if isinstance(wav_input, bytes) else f"file {wav_input}"
        print(f"Error transcribing {input_type}: {e}")
        return None

def transcribe_wav_bytes(wav_bytes):
    """
    Transcribe WAV byte data by writing to a temporary file first
    
    Args:
        wav_bytes (bytes): WAV audio data as bytes
        
    Returns:
        str: Transcription text or None if error
    """
    global app
    
    if app is None:
        print("Loading model for transcription...")
        load_model()
    
    # Create a temporary file to write the WAV bytes
    temp_file = None
    try:
        # Create temporary file with .wav extension
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(wav_bytes)
            temp_file_path = temp_file.name
        
        print(f"Transcribing WAV bytes data ({len(wav_bytes)} bytes) via temp file: {temp_file_path}")
        
        start_time = datetime.now()
        transcription = app.transcribe(temp_file_path)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Transcription completed in {duration:.2f} seconds")
        print(f"Result: {transcription}")
        return transcription
        
    except Exception as e:
        print(f"Error transcribing WAV bytes data: {e}")
        return None
    finally:
        # Clean up temporary file
        if temp_file is not None:
            try:
                os.unlink(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary file {temp_file_path}: {cleanup_error}")

def main():
    """Main function for interactive control"""
    print("Whisper NPU Benchmark Tool")
    print("Commands:")
    print("  start - Start the transcription benchmark")
    print("  stop  - Stop the transcription benchmark")
    print("  status - Check benchmark status")
    print("  transcribe <file> - Transcribe a specific WAV file")
    print("  quit  - Exit the program")
    print()
    
    while True:
        try:
            command = input("Enter command: ").strip()
            command_parts = command.split()
            command_lower = command_parts[0].lower() if command_parts else ""
            
            if command_lower == "start":
                start_benchmark()
            elif command_lower == "stop":
                stop_benchmark()
            elif command_lower == "status":
                if is_running():
                    print(f"Benchmark is running. Transcriptions completed: {get_transcription_count()}")
                else:
                    print("Benchmark is not running.")
            elif command_lower == "transcribe":
                if len(command_parts) < 2:
                    print("Usage: transcribe <wav_file_path>")
                else:
                    wav_file = " ".join(command_parts[1:])  # Support file paths with spaces
                    transcribe_wav(wav_file)
            elif command_lower in ["quit", "exit", "q"]:
                if is_running():
                    stop_benchmark()
                print("Goodbye!")
                break
            else:
                print("Unknown command. Use: start, stop, status, transcribe <file>, or quit")
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
            if is_running():
                stop_benchmark()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

