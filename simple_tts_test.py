#!/usr/bin/env python3
"""
Simple TTS Service Test Script
=============================

A lightweight testing script for quick TTS service validation.
Perfect for development and debugging.

Usage:
    python simple_tts_test.py
    python simple_tts_test.py --text "Custom text to test"
    python simple_tts_test.py --language zh-cn --text "你好世界"
    python simple_tts_test.py --reference-audio ./my_voice.wav
"""

import grpc
import os
import time
import argparse
import logging
from proto import model_service_pb2
from proto import model_service_pb2_grpc

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_tts_service(
    server_address: str = "localhost:50051",
    text: str = "Hello, this is a simple test of the TTS service.",
    language: str = "en",
    reference_audio_path: str = None,
    output_file: str = "simple_test_output.wav",
    timeout: int = 60
):
    """
    Test the TTS service with a simple request
    
    Args:
        server_address: gRPC server address
        text: Text to convert to speech
        language: Language code (en, zh-cn, ja, etc.)
        reference_audio_path: Path to reference audio file (optional)
        output_file: Output WAV file path
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if test successful, False otherwise
    """
    logger = setup_logging()
    
    # Load reference audio if provided
    reference_audio_bytes = None
    if reference_audio_path:
        try:
            with open(reference_audio_path, "rb") as f:
                reference_audio_bytes = f.read()
            logger.info(f"📁 Loaded reference audio: {reference_audio_path}")
        except FileNotFoundError:
            logger.error(f"❌ Reference audio file not found: {reference_audio_path}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading reference audio: {e}")
            return False
    
    try:
        # Connect to gRPC server
        logger.info(f"🔌 Connecting to TTS service at {server_address}...")
        channel = grpc.insecure_channel(server_address)
        
        # Test connection
        try:
            grpc.channel_ready_future(channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            logger.error(f"❌ Cannot connect to server: {server_address}")
            logger.error("   Make sure the server is running!")
            return False
        
        stub = model_service_pb2_grpc.MediaServiceStub(channel)
        logger.info("✅ Connected successfully!")
        
        # Create TTS request
        logger.info(f"🎯 Preparing TTS request...")
        logger.info(f"   Text: '{text}'")
        logger.info(f"   Language: {language}")
        logger.info(f"   Reference Audio: {'Yes' if reference_audio_bytes else 'Default server voice'}")
        
        request = model_service_pb2.TtsRequest(
            text_to_speak=text,
            language=language
        )
        
        if reference_audio_bytes:
            request.reference_audio = reference_audio_bytes
        
        # Send request and measure time
        logger.info("🚀 Sending TTS request...")
        start_time = time.time()
        
        response = stub.Tts(request, timeout=timeout)
        
        duration = time.time() - start_time
        logger.info(f"⏱️  Request completed in {duration:.2f} seconds")
        
        # Validate response
        if not response or not response.generated_audio:
            logger.error("❌ No audio data received in response")
            return False
        
        # Save audio file
        logger.info(f"💾 Saving audio to: {output_file}")
        with open(output_file, "wb") as f:
            f.write(response.generated_audio)
        
        # Calculate audio metrics
        audio_size = len(response.generated_audio)
        logger.info(f"📊 Audio file size: {audio_size:,} bytes ({audio_size/1024:.1f} KB)")
        
        # Try to get audio duration (requires wave module)
        try:
            import wave
            import io
            with wave.open(io.BytesIO(response.generated_audio), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                audio_duration = frames / float(sample_rate)
                logger.info(f"🎵 Audio duration: {audio_duration:.2f} seconds")
                logger.info(f"📈 Generation speed: {audio_duration/duration:.2f}x realtime")
        except Exception:
            logger.info("📊 Audio duration: Could not determine")
        
        logger.info("✅ TTS test completed successfully!")
        logger.info(f"🎧 You can play the generated audio: {output_file}")
        
        return True
        
    except grpc.RpcError as e:
        logger.error(f"❌ gRPC Error occurred:")
        logger.error(f"   Status Code: {e.code()}")
        logger.error(f"   Details: {e.details()}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False
        
    finally:
        try:
            channel.close()
        except:
            pass

def run_quick_tests():
    """Run a series of quick tests with different languages"""
    logger = setup_logging()
    
    logger.info("\n" + "="*60)
    logger.info("🚀 RUNNING QUICK MULTI-LANGUAGE TESTS")
    logger.info("="*60)
    
    test_cases = [
        ("en", "Hello, this is an English test."),
        ("zh-cn", "你好，这是中文测试。"),
        ("ja", "こんにちは、これは日本語のテストです。"),
        ("de", "Hallo, das ist ein deutscher Test."),
    ]
    
    results = []
    for i, (lang, text) in enumerate(test_cases, 1):
        logger.info(f"\n--- Test {i}/{len(test_cases)}: {lang} ---")
        output_file = f"quick_test_{lang}.wav"
        success = test_tts_service(
            text=text,
            language=lang,
            output_file=output_file,
            timeout=30
        )
        results.append((lang, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 QUICK TESTS SUMMARY")
    logger.info("="*60)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for lang, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"   {lang}: {status}")
    
    logger.info(f"\nSuccess Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    return successful == total

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Simple TTS Service Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --text "Custom text to test"
  %(prog)s --language zh-cn --text "你好世界"
  %(prog)s --reference-audio ./my_voice.wav
  %(prog)s --quick-tests
        """
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a simple test of the TTS service.",
        help="Text to convert to speech"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh-cn", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"],
        help="Language code (default: en)"
    )
    
    parser.add_argument(
        "--reference-audio",
        type=str,
        help="Path to reference audio file for voice cloning"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="simple_test_output.wav",
        help="Output WAV file path (default: simple_test_output.wav)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--quick-tests",
        action="store_true",
        help="Run quick tests with multiple languages"
    )
    
    args = parser.parse_args()
    
    if args.quick_tests:
        success = run_quick_tests()
        return 0 if success else 1
    else:
        success = test_tts_service(
            server_address=args.server,
            text=args.text,
            language=args.language,
            reference_audio_path=args.reference_audio,
            output_file=args.output,
            timeout=args.timeout
        )
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())
