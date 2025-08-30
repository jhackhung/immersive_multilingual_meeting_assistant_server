# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
from datetime import datetime
import threading
import time

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

def main():
    """Main function for interactive control"""
    print("Whisper NPU Benchmark Tool")
    print("Commands:")
    print("  start - Start the transcription benchmark")
    print("  stop  - Stop the transcription benchmark")
    print("  status - Check benchmark status")
    print("  quit  - Exit the program")
    print()
    
    while True:
        try:
            command = input("Enter command: ").strip().lower()
            
            if command == "start":
                start_benchmark()
            elif command == "stop":
                stop_benchmark()
            elif command == "status":
                if is_running():
                    print(f"Benchmark is running. Transcriptions completed: {get_transcription_count()}")
                else:
                    print("Benchmark is not running.")
            elif command in ["quit", "exit", "q"]:
                if is_running():
                    stop_benchmark()
                print("Goodbye!")
                break
            else:
                print("Unknown command. Use: start, stop, status, or quit")
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
            if is_running():
                stop_benchmark()
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

