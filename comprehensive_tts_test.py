"""
Comprehensive TTS Service Test Script
====================================

This script provides extensive testing capabilities for the TTS (Text-to-Speech) service,
including performance testing, error handling, and multi-language support.

Features:
- Multi-language testing
- Performance benchmarking
- Error condition testing
- Custom reference audio testing
- Concurrent request testing
- Output validation
"""

import grpc
import os
import time
import argparse
import logging
import json
import threading
import hashlib
import wave
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import generated modules from .proto
from proto import model_service_pb2
from proto import model_service_pb2_grpc

# =====================================================
# Configuration and Test Data
# =====================================================

@dataclass
class TestResult:
    """Data class to store test results"""
    test_name: str
    language: str
    text: str
    success: bool
    duration: float
    output_file: str
    error_message: str = ""
    audio_duration: float = 0.0
    file_size: int = 0

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    total_audio_generated: float = 0.0

# Comprehensive test cases for different languages
LANGUAGE_TEST_CASES = {
    "en": [
        "Hello, this is a comprehensive test of the text-to-speech API in English.",
        "The quick brown fox jumps over the lazy dog.",
        "Welcome to our multilingual meeting assistant!",
        "Testing various sentence structures and punctuation marks: Is this working? Yes, it is!"
    ],
    "zh-cn": [
        "你好，这是一个全面的中文文本转语音API测试。",
        "我们正在测试中文语音合成的质量和性能。",
        "欢迎使用我们的多语言会议助手！",
        "测试各种句式和标点符号：这样工作吗？是的，很好！"
    ],
    "ja": [
        "こんにちは、これは日本語のテキスト読み上げAPIの包括的なテストです。",
        "私たちは日本語音声合成の品質と性能をテストしています。",
        "多言語会議アシスタントへようこそ！",
        "様々な文構造と句読点をテストします：これは機能していますか？はい、機能しています！"
    ],
    "ko": [
        "안녕하세요, 이것은 한국어 텍스트 음성 변환 API의 포괄적인 테스트입니다.",
        "우리는 한국어 음성 합성의 품질과 성능을 테스트하고 있습니다.",
        "다국어 회의 도우미에 오신 것을 환영합니다!",
        "다양한 문장 구조와 구두점을 테스트합니다: 이것이 작동하나요? 네, 작동합니다!"
    ],
    "fr": [
        "Bonjour, ceci est un test complet de l'API de synthèse vocale en français.",
        "Nous testons la qualité et les performances de la synthèse vocale française.",
        "Bienvenue dans notre assistant de réunion multilingue!",
        "Test de diverses structures de phrases et signes de ponctuation: Cela fonctionne-t-il? Oui, ça marche!"
    ],
    "de": [
        "Hallo, dies ist ein umfassender Test der Text-to-Speech-API auf Deutsch.",
        "Wir testen die Qualität und Leistung der deutschen Sprachsynthese.",
        "Willkommen bei unserem mehrsprachigen Meeting-Assistenten!",
        "Test verschiedener Satzstrukturen und Satzzeichen: Funktioniert das? Ja, es funktioniert!"
    ],
    "es": [
        "Hola, esta es una prueba integral de la API de texto a voz en español.",
        "Estamos probando la calidad y el rendimiento de la síntesis de voz en español.",
        "¡Bienvenido a nuestro asistente de reuniones multilingüe!",
        "Prueba de varias estructuras de oraciones y signos de puntuación: ¿Esto funciona? ¡Sí, funciona!"
    ]
}

# Error test cases
ERROR_TEST_CASES = [
    {"name": "Empty text", "text": "", "language": "en", "expected_error": True},
    {"name": "Too long text", "text": "a" * 1001, "language": "en", "expected_error": True},
    {"name": "Unsupported language", "text": "Test text", "language": "xyz", "expected_error": True},
    {"name": "Only whitespace", "text": "   \n\t   ", "language": "en", "expected_error": True},
    {"name": "Special characters only", "text": "!@#$%^&*()", "language": "en", "expected_error": False},
    {"name": "Numbers only", "text": "12345678901", "language": "en", "expected_error": False},
]

class TTSTestClient:
    """Comprehensive TTS testing client"""
    
    def __init__(self, server_address: str = "localhost:50051", timeout: int = 120):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self.results: List[TestResult] = []
        self.metrics = PerformanceMetrics()
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'tts_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Establish connection to gRPC server"""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self.channel).result(timeout=10)
            self.stub = model_service_pb2_grpc.MediaServiceStub(self.channel)
            self.logger.info(f"✅ Successfully connected to gRPC server: {self.server_address}")
            return True
        except grpc.FutureTimeoutError:
            self.logger.error(f"❌ Cannot connect to gRPC server: {self.server_address}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Connection error: {e}")
            return False
            
    def disconnect(self):
        """Close the gRPC connection"""
        if self.channel:
            self.channel.close()
            self.logger.info("🔌 Disconnected from gRPC server")
            
    def get_audio_duration(self, wav_bytes: bytes) -> float:
        """Get duration of audio from WAV bytes"""
        try:
            import io
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            self.logger.warning(f"Could not determine audio duration: {e}")
            return 0.0
            
    def generate_file_hash(self, content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.md5(content).hexdigest()[:8]
        
    def run_single_tts_test(self, 
                           test_name: str, 
                           language: str, 
                           text: str, 
                           output_dir: str,
                           reference_audio_bytes: Optional[bytes] = None,
                           expected_error: bool = False) -> TestResult:
        """Run a single TTS test"""
        
        start_time = time.time()
        success = False
        error_message = ""
        output_file = ""
        audio_duration = 0.0
        file_size = 0
        
        try:
            # Create request
            request = model_service_pb2.TtsRequest(
                text_to_speak=text,
                language=language
            )
            if reference_audio_bytes:
                request.reference_audio = reference_audio_bytes
                
            self.logger.info(f"🎯 Testing: {test_name} | Language: {language} | Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Send gRPC request
            response = self.stub.Tts(request, timeout=self.timeout)
            
            if response and response.generated_audio:
                # Generate unique filename
                file_hash = self.generate_file_hash(response.generated_audio)
                output_file = os.path.join(output_dir, f"{test_name.replace(' ', '_')}_{language}_{file_hash}.wav")
                
                # Save audio file
                with open(output_file, "wb") as f:
                    f.write(response.generated_audio)
                
                # Get audio metrics
                audio_duration = self.get_audio_duration(response.generated_audio)
                file_size = len(response.generated_audio)
                
                success = not expected_error
                if expected_error:
                    error_message = "Expected error but request succeeded"
                    
            else:
                error_message = "No audio data received"
                
        except grpc.RpcError as e:
            if expected_error:
                success = True
                self.logger.info(f"✅ Expected gRPC error received: {e.code()} - {e.details()}")
            else:
                error_message = f"gRPC Error: {e.code()} - {e.details()}"
                self.logger.error(f"❌ Unexpected gRPC error: {error_message}")
                
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            self.logger.error(f"❌ Unexpected error: {error_message}")
            
        duration = time.time() - start_time
        
        # Create result
        result = TestResult(
            test_name=test_name,
            language=language,
            text=text,
            success=success,
            duration=duration,
            output_file=output_file,
            error_message=error_message,
            audio_duration=audio_duration,
            file_size=file_size
        )
        
        # Update metrics
        self.update_metrics(result)
        self.results.append(result)
        
        # Log result
        status = "✅" if success else "❌"
        self.logger.info(f"{status} {test_name} completed in {duration:.2f}s | Audio: {audio_duration:.2f}s | Size: {file_size} bytes")
        
        return result
        
    def update_metrics(self, result: TestResult):
        """Update performance metrics"""
        self.metrics.total_requests += 1
        
        if result.success:
            self.metrics.successful_requests += 1
            self.metrics.total_audio_generated += result.audio_duration
        else:
            self.metrics.failed_requests += 1
            
        self.metrics.min_duration = min(self.metrics.min_duration, result.duration)
        self.metrics.max_duration = max(self.metrics.max_duration, result.duration)
        
        # Calculate average duration
        total_duration = sum(r.duration for r in self.results)
        self.metrics.average_duration = total_duration / len(self.results)
        
    def run_language_tests(self, output_dir: str, reference_audio_bytes: Optional[bytes] = None):
        """Run comprehensive language tests"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🌍 RUNNING MULTI-LANGUAGE TESTS")
        self.logger.info("="*80)
        
        for language, texts in LANGUAGE_TEST_CASES.items():
            for i, text in enumerate(texts):
                test_name = f"Language_Test_{language}_{i+1}"
                self.run_single_tts_test(
                    test_name=test_name,
                    language=language,
                    text=text,
                    output_dir=output_dir,
                    reference_audio_bytes=reference_audio_bytes
                )
                time.sleep(0.5)  # Brief pause between requests
                
    def run_error_tests(self, output_dir: str):
        """Run error condition tests"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🚨 RUNNING ERROR CONDITION TESTS")
        self.logger.info("="*80)
        
        for error_case in ERROR_TEST_CASES:
            self.run_single_tts_test(
                test_name=f"Error_Test_{error_case['name'].replace(' ', '_')}",
                language=error_case['language'],
                text=error_case['text'],
                output_dir=output_dir,
                expected_error=error_case['expected_error']
            )
            time.sleep(0.2)
            
    def run_large_audio_test(self, output_dir: str):
        """Test with large reference audio file"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📁 RUNNING LARGE AUDIO FILE TEST")
        self.logger.info("="*80)
        
        # Generate 11MB of random audio data
        large_audio_bytes = os.urandom(11 * 1024 * 1024)
        
        self.run_single_tts_test(
            test_name="Large_Audio_File_Test",
            language="en",
            text="Testing with oversized reference audio file.",
            output_dir=output_dir,
            reference_audio_bytes=large_audio_bytes,
            expected_error=True
        )
        
    def run_concurrent_tests(self, output_dir: str, num_threads: int = 3):
        """Run concurrent requests to test thread safety"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🔄 RUNNING CONCURRENT TESTS ({num_threads} threads)")
        self.logger.info("="*80)
        
        def worker(thread_id: int):
            test_name = f"Concurrent_Test_Thread_{thread_id}"
            return self.run_single_tts_test(
                test_name=test_name,
                language="en",
                text=f"This is concurrent test number {thread_id} running simultaneously.",
                output_dir=output_dir
            )
            
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)
        
        concurrent_duration = time.time() - start_time
        self.logger.info(f"🏁 Concurrent tests completed in {concurrent_duration:.2f}s")
        
    def run_performance_benchmark(self, output_dir: str, iterations: int = 5):
        """Run performance benchmark tests"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"⚡ RUNNING PERFORMANCE BENCHMARK ({iterations} iterations)")
        self.logger.info("="*80)
        
        benchmark_text = "Performance benchmark test with consistent text for accurate timing measurements."
        
        for i in range(iterations):
            test_name = f"Performance_Benchmark_{i+1}"
            self.run_single_tts_test(
                test_name=test_name,
                language="en",
                text=benchmark_text,
                output_dir=output_dir
            )
            time.sleep(0.1)
            
    def generate_report(self, output_dir: str):
        """Generate comprehensive test report"""
        report_file = os.path.join(output_dir, f"tts_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Prepare report data
        report_data = {
            "test_summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "success_rate": (sum(1 for r in self.results if r.success) / len(self.results) * 100) if self.results else 0
            },
            "performance_metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "average_duration": self.metrics.average_duration,
                "min_duration": self.metrics.min_duration if self.metrics.min_duration != float('inf') else 0,
                "max_duration": self.metrics.max_duration,
                "total_audio_generated_seconds": self.metrics.total_audio_generated
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "language": r.language,
                    "text_length": len(r.text),
                    "success": r.success,
                    "duration": r.duration,
                    "audio_duration": r.audio_duration,
                    "file_size": r.file_size,
                    "output_file": r.output_file,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        # Print summary
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 TEST SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total Tests: {report_data['test_summary']['total_tests']}")
        self.logger.info(f"Successful: {report_data['test_summary']['successful_tests']}")
        self.logger.info(f"Failed: {report_data['test_summary']['failed_tests']}")
        self.logger.info(f"Success Rate: {report_data['test_summary']['success_rate']:.1f}%")
        self.logger.info(f"Average Duration: {report_data['performance_metrics']['average_duration']:.2f}s")
        self.logger.info(f"Total Audio Generated: {report_data['performance_metrics']['total_audio_generated_seconds']:.2f}s")
        self.logger.info(f"Report saved to: {report_file}")
        
        return report_file

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Comprehensive TTS gRPC Service Test Client")
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="gRPC server address and port"
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Reference .wav file path for voice cloning (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tts_comprehensive_test_output",
        help="Output directory for generated audio files"
    )
    parser.add_argument(
        "--test-suite",
        type=str,
        choices=["all", "language", "error", "performance", "concurrent"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--concurrent-threads",
        type=int,
        default=3,
        help="Number of concurrent threads for concurrent tests"
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=5,
        help="Number of iterations for performance benchmark"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reference audio if provided
    reference_audio_bytes = None
    if args.reference_audio:
        try:
            with open(args.reference_audio, "rb") as f:
                reference_audio_bytes = f.read()
            print(f"🔊 Loaded reference audio: {args.reference_audio}")
        except FileNotFoundError:
            print(f"❌ Reference audio file not found: {args.reference_audio}")
            print("Will test with default server voice only.")
    
    # Initialize test client
    client = TTSTestClient(server_address=args.server, timeout=args.timeout)
    
    if not client.connect():
        return 1
        
    try:
        # Run selected test suites
        if args.test_suite in ["all", "language"]:
            client.run_language_tests(args.output_dir, reference_audio_bytes)
            
        if args.test_suite in ["all", "error"]:
            client.run_error_tests(args.output_dir)
            client.run_large_audio_test(args.output_dir)
            
        if args.test_suite in ["all", "performance"]:
            client.run_performance_benchmark(args.output_dir, args.benchmark_iterations)
            
        if args.test_suite in ["all", "concurrent"]:
            client.run_concurrent_tests(args.output_dir, args.concurrent_threads)
            
        # Generate report
        report_file = client.generate_report(args.output_dir)
        print(f"\n📝 Detailed report generated: {report_file}")
        
    finally:
        client.disconnect()
        
    return 0

if __name__ == '__main__':
    exit(main())
