#!/usr/bin/env python3
"""
TTS Voice Cloning Test Script
============================

This script tests the voice cloning capabilities of the TTS service
by using different reference audio files and comparing the results.

Features:
- Test with multiple reference audio files
- Compare voice cloning quality
- Generate side-by-side comparisons
- Validate audio file formats
"""

import grpc
import os
import time
import argparse
import logging
import wave
import io
from typing import List, Dict, Optional
from dataclasses import dataclass

from proto import model_service_pb2
from proto import model_service_pb2_grpc

@dataclass
class VoiceTestResult:
    """Result of a voice cloning test"""
    reference_file: str
    output_file: str
    success: bool
    duration: float
    audio_duration: float
    file_size: int
    error_message: str = ""

class VoiceCloningTester:
    """Voice cloning test client"""
    
    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.logger = self._setup_logging()
        self.channel = None
        self.stub = None
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Connect to gRPC server"""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self.channel).result(timeout=10)
            self.stub = model_service_pb2_grpc.MediaServiceStub(self.channel)
            self.logger.info(f"✅ Connected to TTS service at {self.server_address}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from server"""
        if self.channel:
            self.channel.close()
            
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate that the audio file is a proper WAV file"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                duration = wav_file.getnframes() / sample_rate
                
                self.logger.info(f"📊 Audio info for {os.path.basename(file_path)}:")
                self.logger.info(f"   Channels: {channels}")
                self.logger.info(f"   Sample Rate: {sample_rate} Hz")
                self.logger.info(f"   Duration: {duration:.2f} seconds")
                
                # Basic validation
                if sample_rate < 8000 or sample_rate > 48000:
                    self.logger.warning(f"⚠️  Unusual sample rate: {sample_rate} Hz")
                    
                if duration < 1.0:
                    self.logger.warning(f"⚠️  Very short audio: {duration:.2f} seconds")
                elif duration > 30.0:
                    self.logger.warning(f"⚠️  Very long audio: {duration:.2f} seconds")
                    
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Invalid audio file {file_path}: {e}")
            return False
            
    def test_voice_cloning(self, 
                          reference_audio_path: str, 
                          test_text: str,
                          language: str,
                          output_dir: str) -> VoiceTestResult:
        """Test voice cloning with a specific reference audio file"""
        
        start_time = time.time()
        reference_name = os.path.splitext(os.path.basename(reference_audio_path))[0]
        output_file = os.path.join(output_dir, f"cloned_voice_{reference_name}_{language}.wav")
        
        try:
            # Load and validate reference audio
            self.logger.info(f"🎤 Testing voice cloning with: {reference_audio_path}")
            
            if not self.validate_audio_file(reference_audio_path):
                return VoiceTestResult(
                    reference_file=reference_audio_path,
                    output_file="",
                    success=False,
                    duration=0,
                    audio_duration=0,
                    file_size=0,
                    error_message="Invalid reference audio file"
                )
                
            # Load reference audio bytes
            with open(reference_audio_path, "rb") as f:
                reference_audio_bytes = f.read()
                
            # Create TTS request
            request = model_service_pb2.TtsRequest(
                text_to_speak=test_text,
                language=language,
                reference_audio=reference_audio_bytes
            )
            
            self.logger.info(f"🚀 Generating speech with cloned voice...")
            self.logger.info(f"   Text: '{test_text[:50]}{'...' if len(test_text) > 50 else ''}'")
            self.logger.info(f"   Language: {language}")
            
            # Send request
            response = self.stub.Tts(request, timeout=120)
            
            if not response or not response.generated_audio:
                return VoiceTestResult(
                    reference_file=reference_audio_path,
                    output_file="",
                    success=False,
                    duration=time.time() - start_time,
                    audio_duration=0,
                    file_size=0,
                    error_message="No audio data received"
                )
                
            # Save generated audio
            with open(output_file, "wb") as f:
                f.write(response.generated_audio)
                
            # Calculate metrics
            duration = time.time() - start_time
            file_size = len(response.generated_audio)
            
            # Get audio duration
            audio_duration = 0.0
            try:
                with wave.open(io.BytesIO(response.generated_audio), 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    audio_duration = frames / float(sample_rate)
            except Exception:
                pass
                
            self.logger.info(f"✅ Voice cloning successful!")
            self.logger.info(f"   Generation time: {duration:.2f}s")
            self.logger.info(f"   Audio duration: {audio_duration:.2f}s")
            self.logger.info(f"   Output file: {output_file}")
            
            return VoiceTestResult(
                reference_file=reference_audio_path,
                output_file=output_file,
                success=True,
                duration=duration,
                audio_duration=audio_duration,
                file_size=file_size
            )
            
        except grpc.RpcError as e:
            error_msg = f"gRPC Error: {e.code()} - {e.details()}"
            self.logger.error(f"❌ {error_msg}")
            
            return VoiceTestResult(
                reference_file=reference_audio_path,
                output_file="",
                success=False,
                duration=time.time() - start_time,
                audio_duration=0,
                file_size=0,
                error_message=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            return VoiceTestResult(
                reference_file=reference_audio_path,
                output_file="",
                success=False,
                duration=time.time() - start_time,
                audio_duration=0,
                file_size=0,
                error_message=error_msg
            )
            
    def test_multiple_voices(self, 
                           reference_files: List[str], 
                           test_texts: Dict[str, str],
                           output_dir: str) -> List[VoiceTestResult]:
        """Test voice cloning with multiple reference files and languages"""
        
        results = []
        total_tests = len(reference_files) * len(test_texts)
        current_test = 0
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🎭 STARTING VOICE CLONING TESTS")
        self.logger.info("="*80)
        self.logger.info(f"Reference files: {len(reference_files)}")
        self.logger.info(f"Languages/texts: {len(test_texts)}")
        self.logger.info(f"Total tests: {total_tests}")
        
        for ref_file in reference_files:
            if not os.path.exists(ref_file):
                self.logger.error(f"❌ Reference file not found: {ref_file}")
                continue
                
            self.logger.info(f"\n📁 Testing with reference: {os.path.basename(ref_file)}")
            
            for language, text in test_texts.items():
                current_test += 1
                self.logger.info(f"\n--- Test {current_test}/{total_tests} ---")
                
                result = self.test_voice_cloning(ref_file, text, language, output_dir)
                results.append(result)
                
                time.sleep(0.5)  # Brief pause between tests
                
        return results
        
    def generate_comparison_script(self, results: List[VoiceTestResult], output_dir: str):
        """Generate a script to help compare the generated audio files"""
        
        script_content = """#!/bin/bash
# Audio Comparison Script
# This script helps you compare the generated voice clones

echo "🎧 Voice Cloning Test Results Comparison"
echo "========================================"
echo ""

"""
        
        # Group results by language
        by_language = {}
        for result in results:
            if result.success:
                # Extract language from filename
                filename = os.path.basename(result.output_file)
                if '_en.wav' in filename:
                    lang = 'English'
                elif '_zh-cn.wav' in filename:
                    lang = 'Chinese'
                elif '_ja.wav' in filename:
                    lang = 'Japanese'
                elif '_de.wav' in filename:
                    lang = 'German'
                else:
                    lang = 'Unknown'
                    
                if lang not in by_language:
                    by_language[lang] = []
                by_language[lang].append(result)
        
        # Add comparison commands
        for language, lang_results in by_language.items():
            script_content += f'echo "🌍 {language} Voice Clones:"\n'
            for i, result in enumerate(lang_results, 1):
                ref_name = os.path.splitext(os.path.basename(result.reference_file))[0]
                script_content += f'echo "  {i}. {ref_name} -> {os.path.basename(result.output_file)}"\n'
            script_content += 'echo ""\n'
            
        script_content += """
echo "💡 Tips for comparison:"
echo "  - Use audio software like Audacity to compare waveforms"
echo "  - Listen for voice similarity to reference audio"
echo "  - Check for audio quality and naturalness"
echo "  - Compare pronunciation and accent preservation"
echo ""
echo "Generated files are in the output directory."
"""
        
        script_path = os.path.join(output_dir, "compare_voices.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Also create a Windows batch file
        bat_content = script_content.replace('#!/bin/bash', '@echo off').replace('echo "', 'echo ').replace('"', '')
        bat_path = os.path.join(output_dir, "compare_voices.bat")
        with open(bat_path, 'w') as f:
            f.write(bat_content)
            
        self.logger.info(f"📝 Comparison scripts generated:")
        self.logger.info(f"   Unix/Linux: {script_path}")
        self.logger.info(f"   Windows: {bat_path}")
        
    def print_summary(self, results: List[VoiceTestResult]):
        """Print a summary of test results"""
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 VOICE CLONING TEST SUMMARY")
        self.logger.info("="*80)
        
        self.logger.info(f"Total tests: {len(results)}")
        self.logger.info(f"Successful: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")
        self.logger.info(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            avg_duration = sum(r.duration for r in successful) / len(successful)
            avg_audio_duration = sum(r.audio_duration for r in successful) / len(successful)
            total_audio = sum(r.audio_duration for r in successful)
            
            self.logger.info(f"\nPerformance metrics:")
            self.logger.info(f"  Average generation time: {avg_duration:.2f}s")
            self.logger.info(f"  Average audio duration: {avg_audio_duration:.2f}s")
            self.logger.info(f"  Total audio generated: {total_audio:.2f}s")
            
        if failed:
            self.logger.info(f"\n❌ Failed tests:")
            for result in failed:
                ref_name = os.path.basename(result.reference_file)
                self.logger.info(f"   {ref_name}: {result.error_message}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="TTS Voice Cloning Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --reference-dir ./voice_samples/
  %(prog)s --reference-files sample1.wav sample2.wav
  %(prog)s --reference-files sample.wav --languages en zh-cn
        """
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC server address"
    )
    
    parser.add_argument(
        "--reference-files",
        nargs="+",
        help="List of reference audio files to test"
    )
    
    parser.add_argument(
        "--reference-dir",
        type=str,
        help="Directory containing reference audio files (.wav)"
    )
    
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "zh-cn"],
        choices=["en", "zh-cn", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru"],
        help="Languages to test (default: en zh-cn)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="voice_cloning_test_output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Collect reference files
    reference_files = []
    
    if args.reference_files:
        reference_files.extend(args.reference_files)
        
    if args.reference_dir:
        if os.path.exists(args.reference_dir):
            for file in os.listdir(args.reference_dir):
                if file.lower().endswith('.wav'):
                    reference_files.append(os.path.join(args.reference_dir, file))
        else:
            print(f"❌ Reference directory not found: {args.reference_dir}")
            return 1
            
    if not reference_files:
        print("❌ No reference audio files specified!")
        print("Use --reference-files or --reference-dir to specify audio files.")
        return 1
        
    # Create test texts for each language
    test_texts = {
        "en": "Hello, this is a test of voice cloning technology using artificial intelligence.",
        "zh-cn": "你好，这是使用人工智能进行语音克隆技术的测试。",
        "ja": "こんにちは、これは人工知能を使った音声クローン技術のテストです。",
        "ko": "안녕하세요, 이것은 인공지능을 사용한 음성 복제 기술의 테스트입니다.",
        "fr": "Bonjour, ceci est un test de la technologie de clonage vocal utilisant l'intelligence artificielle.",
        "de": "Hallo, dies ist ein Test der Stimmklon-Technologie mit künstlicher Intelligenz.",
        "es": "Hola, esta es una prueba de la tecnología de clonación de voz usando inteligencia artificial.",
        "it": "Ciao, questo è un test della tecnologia di clonazione vocale usando l'intelligenza artificiale.",
        "pt": "Olá, este é um teste da tecnologia de clonagem de voz usando inteligência artificial.",
        "ru": "Привет, это тест технологии клонирования голоса с использованием искусственного интеллекта."
    }
    
    # Filter test texts by requested languages
    filtered_test_texts = {lang: test_texts[lang] for lang in args.languages if lang in test_texts}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = VoiceCloningTester(args.server)
    
    if not tester.connect():
        return 1
        
    try:
        # Run tests
        results = tester.test_multiple_voices(reference_files, filtered_test_texts, args.output_dir)
        
        # Generate comparison script
        tester.generate_comparison_script(results, args.output_dir)
        
        # Print summary
        tester.print_summary(results)
        
        return 0 if any(r.success for r in results) else 1
        
    finally:
        tester.disconnect()

if __name__ == "__main__":
    exit(main())
