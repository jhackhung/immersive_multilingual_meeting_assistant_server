using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IMMA
{
    using NAudio.CoreAudioApi;
    using NAudio.Wave;
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Reflection.PortableExecutable;
    using System.Threading;
    using System.Threading.Tasks;

    // This class is identical to the one in your original code.
    // It can be shared between SystemAudioSegmenter and MicrophoneAudioSegmenter.

    public class MicrophoneAudioSegmenter
    {
        // --- Configuration ---
        // The volume level to be considered silence. A good starting point is 0.02.
        // This may need tuning depending on the microphone's sensitivity and environment noise.
        private const float SilenceThreshold = 0.02f;

        // The duration of silence (in seconds) that triggers a new segment.
        private const int SilenceDurationSeconds = 2;

        // Maximum duration for a segment in seconds before forcing a new segment.
        private const int MaxSegmentDurationSeconds = 20;

        // --- State Variables ---
        private WasapiCapture? _capture; // Changed to WasapiCapture for microphone input
        private Segment? _currentSegment;
        private int _segmentCount = 0;
        private DateTime _lastAudioTime = DateTime.MinValue;
        private bool _isSilent = false;
        private string _outputDirectory;
        public event Action<Segment>? SegmentRecorded;
        public event Action<WaveInEventArgs>? DataAvailable;
        private WaveFileWriter? fullAudio;
        public WaveFormat? Format => _capture?.WaveFormat;

        public MicrophoneAudioSegmenter(string outputDirectory)
        {
            _outputDirectory = outputDirectory;
            Directory.CreateDirectory(_outputDirectory); // Ensure the directory exists
        }

        public void StartCapture()
        {
            Console.WriteLine("Starting microphone capture...");
            // Initialize microphone capture using the default recording device
            _capture = new WasapiCapture();

            // Subscribe to the event that fires whenever new audio data is available
            _capture.DataAvailable += OnDataAvailable;

            // Create a writer for the full, continuous audio stream.
            // The WaveFormat is taken directly from the capture device.
            fullAudio = new WaveFileWriter(Path.Combine(_outputDirectory, "full_mic_audio.wav"), _capture.WaveFormat);

            // Start recording
            _capture.StartRecording();

            Console.WriteLine("Microphone capture started. Press Enter to stop.");
        }

        public void StopCapture()
        {
            Console.WriteLine("Stopping microphone capture...");
            // Stop recording
            _capture?.StopRecording();

            // Finalize the last segment if it exists
            FinalizeCurrentSegment();

            // Clean up the full audio writer
            fullAudio?.Flush();
            fullAudio?.Close();
            fullAudio?.Dispose();
            fullAudio = null;

            // Clean up capture resources
            if (_capture != null)
            {
                _capture.DataAvailable -= OnDataAvailable;
                _capture.Dispose();
                _capture = null;
            }
            Console.WriteLine("Microphone capture stopped.");
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            DataAvailable?.Invoke(e); // Notify any listeners that data is available
            fullAudio?.Write(e.Buffer, 0, e.BytesRecorded);

            var buffer = e.Buffer;
            int bytesRecorded = e.BytesRecorded;
            bool isCurrentlySilent = true;

            // Create a WaveBuffer to easily iterate over audio samples.
            // WasapiCapture typically provides 32-bit float samples, which is what this loop expects.
            var waveBuffer = new WaveBuffer(buffer);
            for (int i = 0; i < bytesRecorded / 4; i++)
            {
                // The absolute value of the sample's amplitude
                float sampleAmplitude = Math.Abs(waveBuffer.FloatBuffer[i]);
                if (sampleAmplitude > SilenceThreshold)
                {
                    isCurrentlySilent = false;
                    break; // Exit early if sound is detected
                }
            }

            // Check if current segment has exceeded max duration
            if (_currentSegment != null && (DateTime.Now - _currentSegment.StartTime).TotalSeconds >= MaxSegmentDurationSeconds)
            {
                Console.WriteLine($"Maximum segment duration of {MaxSegmentDurationSeconds} seconds reached, finalizing segment.");
                FinalizeCurrentSegment();
            }

            if (!isCurrentlySilent)
            {
                // If audio is detected, start a new segment if we were previously silent or have no writer
                if (_currentSegment == null)
                {
                    StartNewSegment();
                }

                // Write the audio data to the current segment file
                _currentSegment?.Writer.Write(buffer, 0, bytesRecorded);
                _lastAudioTime = DateTime.Now; // Update the time we last heard sound
                _isSilent = false;
            }
            else // We detected only silence in this buffer
            {
                // If we are currently recording and the silence has lasted long enough, end the segment
                if (!_isSilent && _currentSegment != null && (DateTime.Now - _lastAudioTime).TotalSeconds >= SilenceDurationSeconds)
                {
                    Console.WriteLine("Silence detected, finalizing segment.");
                    FinalizeCurrentSegment();
                    _isSilent = true;
                }
            }
        }

        private void StartNewSegment()
        {
            // Guard against starting a segment if capture isn't active
            if (_capture == null) return;

            _segmentCount++;
            string segmentPath = Path.Combine(_outputDirectory, $"mic_segment_{_segmentCount}.wav");
            Console.WriteLine($"Starting new segment: {segmentPath}");

            _currentSegment = new()
            {
                Writer = new WaveFileWriter(segmentPath, _capture.WaveFormat),
                StartTime = DateTime.Now
            };
        }

        private void FinalizeCurrentSegment()
        {
            if (_currentSegment != null)
            {
                var filename = _currentSegment.Writer.Filename;
                Console.WriteLine($"Finalizing segment: {filename}");
                _currentSegment.EndTime = DateTime.Now;
                _currentSegment.Writer.Flush();
                _currentSegment.Writer.Close();
                _currentSegment.Writer.Dispose();
                SegmentRecorded?.Invoke(_currentSegment); // Notify listeners about the new segment
                _currentSegment = null;
            }
        }
    }
}