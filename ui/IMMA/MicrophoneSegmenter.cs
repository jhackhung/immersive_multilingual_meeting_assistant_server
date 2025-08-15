using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IMMA
{
    public class MicrophoneSegmenter
    {
        // --- Configuration ---
        // The volume threshold for detecting speech. This is for 16-bit audio, which has a range
        // from -32768 to 32767. A value of 500 is a good starting point for typical speech.
        private const int SilenceThreshold = 500;

        // The duration of silence (in seconds) that finalizes a speaking segment.
        private const int SilenceDurationSeconds = 2;

        // --- State Variables ---
        private WaveInEvent? _waveSource;
        private Segment? _currentSegment;
        private int _segmentCount = 0;
        private DateTime _lastAudioTime = DateTime.MinValue;
        private bool _isSilent = true;
        private readonly string _outputDirectory;
        public event Action<WaveInEventArgs>? DataAvailable;
        public event Action<Segment>? SegmentRecorded;
        public readonly WaveFormat Format = new WaveFormat(16000, 1);
        public MicrophoneSegmenter(string outputDirectory)
        {
            _outputDirectory = outputDirectory;
            Directory.CreateDirectory(_outputDirectory); // Ensure the output directory exists
        }

        public void StartCapture()
        {
            Console.WriteLine("Starting to listen for speech...");
            _waveSource = new WaveInEvent
            {
                DeviceNumber = 0, // Default microphone
                WaveFormat =Format // 16kHz, 16-bit, Mono - common for speech recognition
            };

            _waveSource.DataAvailable += OnDataAvailable;
            _waveSource.StartRecording();

            Console.WriteLine("Listening... Speak into the microphone. Press Enter to stop.");
        }

        public void Stop()
        {
            Console.WriteLine("Stopping listener...");
            _waveSource?.StopRecording();

            FinalizeCurrentSegment(); // Save any segment that was in progress

            if (_waveSource != null)
            {
                _waveSource.DataAvailable -= OnDataAvailable;
                _waveSource.Dispose();
                _waveSource = null;
            }
            Console.WriteLine("Listener stopped.");
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            DataAvailable?.Invoke(e); // Notify any listeners that data is available

            bool soundDetected = false;
            // Iterate through the buffer of 16-bit audio samples
            for (int i = 0; i < e.BytesRecorded; i += 2)
            {
                // Convert two bytes to a 16-bit sample
                short sample = (short)((e.Buffer[i + 1] << 8) | e.Buffer[i]);
                // Check if the amplitude of the sample is above our threshold
                if (Math.Abs(sample) > SilenceThreshold)
                {
                    soundDetected = true;
                    break;
                }
            }

            // Write the audio data to the current segment file
            _currentSegment?.Writer.Write(e.Buffer, 0, e.BytesRecorded);

            if (soundDetected)
            {
                // If sound is detected, start a new segment if we aren't already recording one
                if (_isSilent)
                {
                    _isSilent = false;
                    StartNewSegment();
                }
                _lastAudioTime = DateTime.Now; // Update the time we last heard sound
            }
            else if (!_isSilent && (DateTime.Now - _lastAudioTime).TotalSeconds >= SilenceDurationSeconds)
            {
                // If we are recording and silence has lasted long enough, end the segment
                _isSilent = true;
                Console.WriteLine("Silence detected, finalizing segment.");
                FinalizeCurrentSegment();
            }
        }

        private void StartNewSegment()
        {
            _segmentCount++;
            string segmentPath = Path.Combine(_outputDirectory, $"speech_{_segmentCount}.wav");
            Console.WriteLine($"Sound detected, starting new segment: {segmentPath}");
            _currentSegment = new() { Writer = new WaveFileWriter(segmentPath, _waveSource!.WaveFormat), StartTime = DateTime.Now };
        }

        private void FinalizeCurrentSegment()
        {
            if (_currentSegment != null)
            {
                Console.WriteLine($"Finalizing segment: {_currentSegment.Writer.Filename}");
                _currentSegment.Writer.Dispose();
                SegmentRecorded?.Invoke(_currentSegment); // Notify listeners about the new segment
                _currentSegment = null;
            }
        }
    }

}
