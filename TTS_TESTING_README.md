# TTS Service Testing Suite

This directory contains comprehensive testing scripts for the Text-to-Speech (TTS) service. These scripts help validate functionality, performance, and voice cloning capabilities.

## Test Scripts Overview

### 1. `simple_tts_test.py` - Quick Testing
**Purpose**: Simple, lightweight testing for development and debugging.

**Features**:
- Single request testing
- Quick multi-language validation
- Basic error handling
- Easy-to-use command line interface

**Usage**:
```bash
# Basic test
python simple_tts_test.py

# Custom text and language
python simple_tts_test.py --text "Hello world" --language en

# Test with reference audio
python simple_tts_test.py --reference-audio ./my_voice.wav

# Quick multi-language tests
python simple_tts_test.py --quick-tests
```

### 2. `comprehensive_tts_test.py` - Full Testing Suite
**Purpose**: Comprehensive testing with detailed metrics and reporting.

**Features**:
- Multi-language testing (7+ languages)
- Performance benchmarking
- Error condition testing
- Concurrent request testing
- Detailed JSON reports
- Performance metrics tracking

**Usage**:
```bash
# Run all tests
python comprehensive_tts_test.py

# Run specific test suites
python comprehensive_tts_test.py --test-suite language
python comprehensive_tts_test.py --test-suite performance
python comprehensive_tts_test.py --test-suite concurrent

# Custom configuration
python comprehensive_tts_test.py --concurrent-threads 5 --benchmark-iterations 10
```

### 3. `voice_cloning_test.py` - Voice Cloning Validation
**Purpose**: Specialized testing for voice cloning capabilities.

**Features**:
- Multiple reference audio file testing
- Voice quality validation
- Audio file format checking
- Comparison script generation
- Cross-language voice cloning

**Usage**:
```bash
# Test with specific reference files
python voice_cloning_test.py --reference-files voice1.wav voice2.wav

# Test all WAV files in a directory
python voice_cloning_test.py --reference-dir ./voice_samples/

# Test specific languages
python voice_cloning_test.py --reference-dir ./samples/ --languages en zh-cn ja
```

### 4. `test_tts_service.py` - Original Test Script
**Purpose**: Legacy test script with basic functionality.

**Features**:
- Multi-language testing
- Error condition validation
- Reference audio testing
- Basic performance measurement

## Test Data Structure

### Output Directory Structure
```
test_output/
├── simple_test_output.wav          # Simple test results
├── comprehensive_test_output/       # Comprehensive test results
│   ├── Language_Test_en_1.wav
│   ├── Performance_Benchmark_1.wav
│   └── tts_test_report_YYYYMMDD_HHMMSS.json
├── voice_cloning_test_output/      # Voice cloning results
│   ├── cloned_voice_speaker1_en.wav
│   ├── cloned_voice_speaker1_zh-cn.wav
│   ├── compare_voices.sh
│   └── compare_voices.bat
└── tts_test_output/                # Original test output
    ├── default_speaker_en.wav
    └── custom_speaker_zh-cn.wav
```

## Supported Languages

| Code   | Language           | Test Coverage |
|--------|--------------------|---------------|
| `en`   | English            | ✅ Full       |
| `zh-cn`| Chinese (Simplified)| ✅ Full       |
| `ja`   | Japanese           | ✅ Full       |
| `ko`   | Korean             | ✅ Full       |
| `fr`   | French             | ✅ Full       |
| `de`   | German             | ✅ Full       |
| `es`   | Spanish            | ✅ Full       |
| `it`   | Italian            | ⚠️ Basic     |
| `pt`   | Portuguese         | ⚠️ Basic     |
| `ru`   | Russian            | ⚠️ Basic     |

## Test Categories

### 1. Functional Tests
- **Basic TTS**: Text to speech conversion
- **Multi-language**: Different language support
- **Voice Cloning**: Reference audio usage
- **Error Handling**: Invalid input handling

### 2. Performance Tests
- **Speed**: Generation time measurement
- **Concurrency**: Multiple simultaneous requests
- **Memory**: Resource usage tracking
- **Scalability**: Load testing capabilities

### 3. Quality Tests
- **Audio Quality**: Output validation
- **Voice Similarity**: Clone quality assessment
- **Language Accuracy**: Pronunciation validation
- **Consistency**: Reproducible results

## Common Command Line Options

### Server Configuration
```bash
--server localhost:50051          # gRPC server address
--timeout 120                     # Request timeout (seconds)
```

### Audio Configuration
```bash
--reference-audio ./voice.wav     # Reference audio file
--reference-dir ./audio_samples/  # Reference audio directory
--output-dir ./test_results/      # Output directory
```

### Test Configuration
```bash
--language en                     # Single language
--languages en zh-cn ja          # Multiple languages
--test-suite all                  # Test suite selection
--concurrent-threads 3           # Concurrent test threads
--benchmark-iterations 5         # Performance test iterations
```

## Prerequisites

### Python Dependencies
```bash
pip install grpc grpcio-tools
pip install numpy scipy torchaudio
pip install wave logging argparse
```

### System Requirements
- Python 3.7+
- gRPC service running on target server
- Audio playback capability (for manual verification)

## Running the TTS Service

Before running tests, ensure the TTS service is running:

```bash
# Start the TTS service
python server.py

# Verify service is running
python simple_tts_test.py --text "Test connection"
```

## Test Results Analysis

### Success Metrics
- **Success Rate**: Percentage of successful requests
- **Average Duration**: Mean processing time
- **Audio Quality**: Manual assessment required
- **Error Rate**: Failed request percentage

### Performance Metrics
- **Generation Speed**: Audio duration / processing time
- **Throughput**: Requests per second
- **Memory Usage**: Resource consumption
- **Concurrency**: Parallel request handling

### Report Files
- **JSON Reports**: Detailed test results and metrics
- **Log Files**: Timestamped execution logs
- **Audio Files**: Generated speech samples
- **Comparison Scripts**: Helper tools for analysis

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if service is running
   netstat -an | grep 50051
   # Restart service if needed
   python server.py
   ```

2. **Audio Quality Issues**
   ```bash
   # Test with different reference audio
   python voice_cloning_test.py --reference-files good_quality.wav
   # Check audio file format
   file reference_audio.wav
   ```

3. **Performance Issues**
   ```bash
   # Reduce concurrent threads
   python comprehensive_tts_test.py --concurrent-threads 1
   # Increase timeout
   python comprehensive_tts_test.py --timeout 300
   ```

4. **Memory Issues**
   ```bash
   # Use smaller test suites
   python comprehensive_tts_test.py --test-suite language
   # Avoid large reference audio files
   ```

## Best Practices

### For Development Testing
1. Use `simple_tts_test.py` for quick validation
2. Test with short, simple text first
3. Verify connection before extensive testing

### For Quality Assurance
1. Run `comprehensive_tts_test.py` for full validation
2. Test all supported languages
3. Include error condition testing
4. Document any failures or quality issues

### For Voice Cloning
1. Use high-quality reference audio (22kHz+)
2. Ensure reference audio is 3-10 seconds long
3. Test with multiple speakers
4. Compare results across languages

### For Performance Testing
1. Start with single-threaded tests
2. Gradually increase concurrency
3. Monitor system resources
4. Test with realistic workloads

## Contributing

When adding new test cases:

1. **Follow naming conventions**: `test_description_language.wav`
2. **Update test data**: Add new languages or scenarios
3. **Document changes**: Update this README
4. **Validate backwards compatibility**: Ensure existing tests still work

## Support

For issues with the test scripts:

1. Check the logs for detailed error messages
2. Verify TTS service is running and accessible
3. Ensure all dependencies are installed
4. Test with the simple script first
5. Check audio file formats and quality

---

*Last updated: 2025-08-29*
