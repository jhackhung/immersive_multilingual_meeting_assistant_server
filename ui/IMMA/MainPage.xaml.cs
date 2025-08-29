using CommunityToolkit.Maui.Storage;
using GenerativeAI;
using GenerativeAI.Types;
using Google.Protobuf;
using Grpc.Core;
using Grpc.Net.Client;
using IMMA.Grpc;
using Microsoft.Maui.Controls.Shapes;
using NAudio.Wave;
using System.Linq;
using System.Speech.Recognition;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace IMMA
{
    public class TranscriptEntry
    {
        public DateTime Timestamp { get; set; }
        public DateTime EndTime { get; set; }
        public string SpeakerName { get; set; }
        public string Content { get; set; }
        public string SpeakerRole { get; set; }
        public Color TextColor { get; set; } = Color.FromArgb("#1e40af");

        public string FormattedTimestamp => Timestamp.ToString("HH:mm:ss") + "-" + EndTime.ToString("HH:mm:ss");
        public string DisplayName => string.IsNullOrEmpty(SpeakerRole)
            ? SpeakerName
            : $"{SpeakerName} ({SpeakerRole})";
    }
    public class STTTask
    {
        public SpeechRecognitionRequest Request { get; set; }
        public DateTime StartTime { get; set; }
        public bool IsSystemAudio { get; set; } = false;
    }
    public partial class MainPage : ContentPage
    {
        private readonly List<TranscriptEntry> Transcripts = [];
        private readonly SystemAudioSegmenter SystemAudio;
        private readonly MicrophoneSegmenter Microphone;
        const string SystenSegmentsDirectory = "sys_segments";
        const string MicrpohneSegmentsDirectory = "mic_segments";
        Queue<STTTask> speechRecognitionRequests = new();
        MediaService.MediaServiceClient Client;
        TranslatorService.TranslatorServiceClient Translator;
        DateTime MeetingStartTime;
        GenerativeModel chatModel;
        GoogleAi googleAI;
        public MainPage()
        {
            Client = new MediaService.MediaServiceClient(GrpcChannel.ForAddress("http://127.0.0.1:50051"));
            Translator = new TranslatorService.TranslatorServiceClient(GrpcChannel.ForAddress("http://127.0.0.1:50051"));

            // 1) Initialize your AI instance (GoogleAi) with credentials or environment variables
            googleAI = new GoogleAi(Environment.GetEnvironmentVariable("GEMINI_API_KEY"));

            // 2) Create a GenerativeModel using the model name "gemini-1.5-flash"
            chatModel = googleAI.CreateGenerativeModel("models/gemini-1.5-flash");
            InitializeComponent();


            SystemAudio = new SystemAudioSegmenter(SystenSegmentsDirectory);
            if (Directory.Exists(SystenSegmentsDirectory))
            {
                Directory.Delete(SystenSegmentsDirectory, true);
            }
            Directory.CreateDirectory(SystenSegmentsDirectory);
            SystemAudio.SegmentRecorded += (s) =>
            {
                using var stream = File.OpenRead(s.Writer.Filename);
                var resuet = new SpeechRecognitionRequest()
                {
                    AudioData = Google.Protobuf.ByteString.FromStream(stream),
                    ModelSize = "small",
                    Language = "auto",
                    ReturnTimestamps = true,
                };
                var task = new STTTask()
                {
                    Request = resuet,
                    StartTime = s.StartTime,
                    IsSystemAudio = true
                };
                speechRecognitionRequests.Enqueue(task);
            };



            Microphone = new MicrophoneSegmenter(SystenSegmentsDirectory);
            if (Directory.Exists(MicrpohneSegmentsDirectory))
            {
                Directory.Delete(MicrpohneSegmentsDirectory, true);
            }
            Directory.CreateDirectory(MicrpohneSegmentsDirectory);
            Microphone.SegmentRecorded += (s) =>
            {
                using var stream = File.OpenRead(s.Writer.Filename);
                var resuet = new SpeechRecognitionRequest()
                {
                    AudioData = Google.Protobuf.ByteString.FromStream(stream),
                    ModelSize = "small",
                    Language = "auto",
                    ReturnTimestamps = true,
                };
                var task = new STTTask()
                {
                    Request = resuet,
                    StartTime = s.StartTime,
                    IsSystemAudio = false
                };
                speechRecognitionRequests.Enqueue(task);
            };

            Microphone.DataAvailable += (e) =>
            {
                if (translateRecording)
                {
                    transalteRecordingWriter!.Write(e.Buffer, 0, e.BytesRecorded);
                }
            };

            // Speech recognition worker
            Task.Run(async () =>
            {
                while (true)
                {
                    if (speechRecognitionRequests.TryDequeue(out var task))
                    {

                        var result = await Client.SpeechRecognitionAsync(task.Request);
                        foreach (var seg in result.Segments)
                        {
                            Dispatcher.Dispatch(() =>
                            {
                                var entry = new TranscriptEntry()
                                {
                                    SpeakerName = task.IsSystemAudio ? "Unidentified" : "You",
                                    SpeakerRole = "",
                                    Content = seg.Text,
                                    Timestamp = task.StartTime.AddSeconds(seg.StartTime),
                                    EndTime = task.StartTime.AddSeconds(seg.EndTime),
                                    TextColor = Color.FromArgb("#047857")
                                };
                                Transcripts.Add(entry);
                                AddTranscriptEntryToUI(entry);
                            });
                        }
                    }
                    await Task.Delay(100);
                }
            });


            var tapGestureRecognizer = new TapGestureRecognizer();
            tapGestureRecognizer.Tapped += (s, e) =>
            {
                Clipboard.Default.SetTextAsync(SummaryMarkdown.MarkdownText);
                DisplayAlert("Info", "Summary copied to clipboard", "OK");
            };
            SummaryMarkdown.GestureRecognizers.Add(tapGestureRecognizer);
        }


        private void OnTranscriptTabClicked(object sender, EventArgs e)
        {
            SummaryTab.IsVisible = false;
            TranscriptTab.IsVisible = true;
            TranscriptTabButton.BackgroundColor = Color.FromArgb("3b82f6");
            TranscriptTabButton.TextColor = Colors.White;
            SummaryTabButton.BackgroundColor = Color.FromArgb("e2e8f0");
            SummaryTabButton.TextColor = Color.FromArgb("4a5568");
        }

        private void OnSummaryTabClicked(object sender, EventArgs e)
        {
            TranscriptTab.IsVisible = false;
            SummaryTab.IsVisible = true;
            SummaryTabButton.BackgroundColor = Color.FromArgb("3b82f6");
            SummaryTabButton.TextColor = Colors.White;
            TranscriptTabButton.BackgroundColor = Color.FromArgb("e2e8f0");
            TranscriptTabButton.TextColor = Color.FromArgb("4a5568");
        }

        public void AddTranscriptEntryToUI(TranscriptEntry entry)
        {
            // Create the border container
            var border = new Border
            {
                StrokeThickness = 1,
                StrokeShape = new RoundRectangle { CornerRadius = new CornerRadius(12) },
                Padding = new Thickness(10),
                Margin = new Thickness(5)
            };

            // Create the vertical layout inside the border
            var layout = new VerticalStackLayout();

            // Add timestamp label
            var timestampLabel = new Label
            {
                Text = entry.FormattedTimestamp,
                FontSize = 12,
                TextColor = Color.FromArgb("#64748b")
            };
            layout.Add(timestampLabel);

            // Add speaker name label - make it tappable
            var speakerLabel = new Label
            {
                Text = entry.DisplayName,
                FontAttributes = FontAttributes.Bold,
                TextColor = entry.TextColor
            };

            // Add tap gesture to speaker label for renaming
            var tapGestureRecognizer = new TapGestureRecognizer();
            tapGestureRecognizer.Tapped += async (s, e) => await OnSpeakerLabelTapped(entry);
            speakerLabel.GestureRecognizers.Add(tapGestureRecognizer);

            layout.Add(speakerLabel);

            // Add content label
            var contentLabel = new Label
            {
                Text = entry.Content
            };
            layout.Add(contentLabel);

            // Add the layout to the border
            border.Content = layout;

            // Add the border to the transcript area
            TranscriptArea.Add(border);

            // Optionally scroll to the bottom to show the new entry
            Dispatcher.Dispatch(async () =>
            {
                await Task.Delay(100); // Give UI time to update
                await TranscriptTab.ScrollToAsync(0, TranscriptArea.Height, true);
            });
        }

        // Handle speaker label tap events for renaming
        private async Task OnSpeakerLabelTapped(TranscriptEntry tappedEntry)
        {
            // Don't allow renaming the user's entries
            if (tappedEntry.SpeakerName == "You")
            {
                await DisplayAlert("Info", "Cannot modify your own name", "OK");
                return;
            }

            // Show dialog to input new name
            string result = await DisplayPromptAsync(
                "Rename Speaker",
                $"Please enter new name (Current: {tappedEntry.SpeakerName})",
                "OK",
                "Cancel",
                placeholder: "New name",
                initialValue: tappedEntry.SpeakerName);

            // If user entered a new name
            if (!string.IsNullOrWhiteSpace(result) && result != tappedEntry.SpeakerName)
            {
                // Save the original name for finding matches
                string originalName = tappedEntry.SpeakerName;

                // Update all transcript entries with the same speaker name
                foreach (var entry in Transcripts.Where(t => t.SpeakerName == originalName))
                {
                    entry.SpeakerName = result;
                }

                // Refresh the UI by clearing and rebuilding the transcript area
                TranscriptArea.Clear();
                foreach (var entry in Transcripts)
                {
                    AddTranscriptEntryToUI(entry);
                }
            }
        }

        private async void OnAvatarImageButtonClicked(object sender, EventArgs e)
        {
            try
            {
                // Check if the device supports picking photos
                if (!MediaPicker.Default.IsCaptureSupported)
                {
                    await DisplayAlert("Not Supported", "Photo picking is not supported on this device.", "OK");
                    return;
                }

                // Launch the photo picker
                var result = await MediaPicker.Default.PickPhotoAsync(new MediaPickerOptions
                {
                    Title = "Select Avatar Image"
                });

                // If the user selected an image
                if (result != null)
                {
                    // Create a stream from the selected image
                    using var stream = await result.OpenReadAsync();
                    var imgStream = new MemoryStream();
                    stream.CopyTo(imgStream);
                    // Create an ImageSource from the stream
                    imgData = imgStream.ToArray();
                    var imageSource = Microsoft.Maui.Controls.ImageSource.FromStream(() => new MemoryStream(imgData));

                    // Update the avatar image
                    AvatarImageButton.Source = imageSource;
                    // Optional: Save the image to the app's local storage for persistence
                    // This would require additional code to save the file and load it on app startup
                }
            }
            catch (Exception ex)
            {
                // Handle any exceptions
                await DisplayAlert("Error", $"Error selecting image: {ex.Message}", "OK");
            }
        }


        public static void JoinWavFiles(string outputFile, IEnumerable<string> inputFiles)
        {
            WaveFileWriter? writer = null;

            try
            {
                foreach (string inputFile in inputFiles)
                {
                    using (var reader = new WaveFileReader(inputFile))
                    {
                        // On the first file, create the writer and copy the header
                        if (writer == null)
                        {
                            writer = new WaveFileWriter(outputFile, reader.WaveFormat);
                        }
                        // For subsequent files, ensure the format is the same
                        else
                        {
                            if (!reader.WaveFormat.Equals(writer.WaveFormat))
                            {
                                throw new InvalidOperationException("Cannot join WAV files with different formats.");
                            }
                        }

                        // Simple and efficient way to copy all audio data
                        reader.CopyTo(writer);
                    }
                }
            }
            finally
            {
                // Finalize and close the writer
                writer?.Dispose();
            }
        }


        private async void OnSpeakerRecognitionClicked(object sender, EventArgs e)
        {
            DisableButton(SpeakerRecognitionButton);
            try
            {
                var annoteResult = await Client.SpeakerAnnoteAsync(new SpeakerAnnoteRequest()
                {
                    AudioData = Google.Protobuf.ByteString.FromStream(File.OpenRead(System.IO.Path.Combine(SystenSegmentsDirectory, "full.wav")))
                });

                // Process speaker annotation results
                if (annoteResult.AllSegments.Count > 0)
                {
                    // Clear the transcript area to rebuild with speaker information
                    TranscriptArea.Clear();

                    // Dictionary to store speakers with their colors
                    Dictionary<string, Color> speakerColors = new Dictionary<string, Color>
                    {
                        { "SPEAKER_00", Color.FromArgb("#1e40af") },
                        { "SPEAKER_01", Color.FromArgb("#047857") },
                        { "SPEAKER_02", Color.FromArgb("#7e22ce") },
                        { "SPEAKER_03", Color.FromArgb("#b91c1c") },
                        { "SPEAKER_04", Color.FromArgb("#0e7490") }
                    };

                    // Update transcript entries with speaker information
                    foreach (var segment in annoteResult.AllSegments)
                    {
                        // Find matching transcript entries based on timestamp overlap
                        var startTime = MeetingStartTime.AddSeconds(segment.StartTime);
                        var endTime = MeetingStartTime.AddSeconds(segment.EndTime);

                        var matchingEntries = Transcripts.Where(t =>
                            (t.Timestamp >= startTime && t.Timestamp <= endTime) ||
                            (t.EndTime >= startTime && t.EndTime <= endTime) ||
                            (t.Timestamp <= startTime && t.EndTime >= endTime)).ToList();

                        foreach (var entry in matchingEntries)
                        {
                            // Skip entries that are from the microphone (user's speech)
                            if (entry.SpeakerName == "You")
                                continue;

                            // Update the speaker information
                            entry.SpeakerName = segment.Speaker;

                            // Assign a consistent color for this speaker
                            if (speakerColors.TryGetValue(segment.Speaker, out var color))
                            {
                                entry.TextColor = color;
                            }
                        }
                    }
                    TranscriptArea.Clear();
                    Transcripts.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
                    MergeConsecutiveSpeakerEntries();


                    // Rebuild the transcript area with updated entries
                    foreach (var entry in Transcripts)
                    {
                        AddTranscriptEntryToUI(entry);
                    }

                    await DisplayAlert("Success", "Speaker identification completed and meeting records updated", "OK");
                }
                else
                {
                    await DisplayAlert("Info", "Not enough speaker information detected", "OK");
                }


            }
            catch (RpcException ex)
            {
                await DisplayAlert("Error", $"Speaker identification service error: {ex.Status.Detail}", "OK");
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", $"Error during speaker identification: {ex.Message}", "OK");
            }
            EnableButton(SpeakerRecognitionButton);
        }

        private void OnStopButtonClicked(object sender, EventArgs e)
        {
            SystemAudio.StopCapture();
            Microphone.Stop();
            Microphone.StartCapture();
            DisableButton(StopButton);
            EnableButton(StartButton);
        }

        private void OnStartButtonClicked(object sender, EventArgs e)
        {
            TranscriptArea.Clear();
            Transcripts.Clear();
            MeetingStartTime = DateTime.Now;
            SystemAudio.StartCapture();
            Microphone.StartCapture();
            DisableButton(StartButton);
            EnableButton(StopButton);
        }

        private void MergeConsecutiveSpeakerEntries()
        {
            if (Transcripts.Count <= 1)
                return;

            var mergedTranscripts = new List<TranscriptEntry>();
            TranscriptEntry? currentGroup = null;

            // Sort entries by timestamp to ensure proper sequential processing
            var sortedTranscripts = Transcripts.OrderBy(t => t.Timestamp).ToList();

            foreach (var entry in sortedTranscripts)
            {
                // If this is the first entry or from a different speaker, start a new group
                if (currentGroup == null || entry.SpeakerName != currentGroup.SpeakerName)
                {
                    // Add the previous group to our results if it exists
                    if (currentGroup != null)
                    {
                        mergedTranscripts.Add(currentGroup);
                    }

                    // Start a new group with this entry
                    currentGroup = new TranscriptEntry
                    {
                        SpeakerName = entry.SpeakerName,
                        SpeakerRole = entry.SpeakerRole,
                        Content = entry.Content,
                        Timestamp = entry.Timestamp,
                        EndTime = entry.EndTime,
                        TextColor = entry.TextColor
                    };
                }
                else
                {
                    // Same speaker, merge the content
                    currentGroup.Content += "\n" + entry.Content;
                    // Update the end time if this entry ends later
                    if (entry.EndTime > currentGroup.EndTime)
                    {
                        currentGroup.EndTime = entry.EndTime;
                    }
                }
            }

            // Don't forget to add the last group
            if (currentGroup != null)
            {
                mergedTranscripts.Add(currentGroup);
            }

            // Replace the original transcript list with our merged version
            Transcripts.Clear();
            Transcripts.AddRange(mergedTranscripts);
        }

        private async void OnSummaryButtonClicked(object sender, EventArgs e)
        {
            DisableButton(GenerateSummaryButton);
            try
            {
                var transcript = string.Join("\n", Transcripts.Select(t => $"{t.FormattedTimestamp} {t.SpeakerName}:\n {t.Content}"));
                var example = new MeetingSummary()
                {
                    Title = "Sample Meeting Summary",
                    Points =
                    [
                        new BulletPoint
                        {
                            Title = "Main Topics",
                            Contents = ["New product feature planning and timeline", "Market competition analysis", "Budget allocation"]
                        },
                        new BulletPoint
                        {
                            Title = "Decisions Made",
                            Contents = ["Complete feature specification by next Monday", "Increase R&D budget by 15%", "First feature demo in 3 weeks"]
                        }
                    ]
                };
                var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
                {
                    WriteIndented = true,
                    Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
                };
                var exData = JsonSerializer.Serialize(example, options);
                var chat = chatModel.StartChat();
                var result = await chat.GenerateContentAsync("Generate meeting summary and key points in Markdown format:\n" + transcript);
                if (result is null)
                {
                    await DisplayAlert("Error", "Unable to generate meeting summary", "OK");
                    return;
                }

                for (int i = 0; i < result.Text.Length; i++)
                {
                    SummaryMarkdown.MarkdownText = result.Text.Substring(0, i);
                    await Task.Delay(20);
                }
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", "Unable to generate meeting summary:" + ex.Message, "OK");
            }
            EnableButton(GenerateSummaryButton);
        }

        public void AddSummarySection(string title, string[] bulletPoints)
        {
            // Find the parent VerticalStackLayout in the SummaryTab ScrollView
            var summaryTabContent = (VerticalStackLayout)SummaryTab.Content;

            // Create the border with the same style as in XAML
            var border = new Border
            {
                StrokeShape = new RoundRectangle { CornerRadius = 12 },
                BackgroundColor = Colors.White,
                Padding = new Thickness(10),
                StrokeThickness = 1,
                Margin = new Thickness(0, 5)
            };

            // Create the content layout
            var contentLayout = new VerticalStackLayout
            {
                // Add the title
                new Label
                {
                    Text = title,
                    FontAttributes = FontAttributes.Bold,
                    TextColor = Color.Parse("#10b981")
                }
            };

            // Add bullet points
            foreach (var point in bulletPoints)
            {
                contentLayout.Add(new Label
                {
                    Text = "• " + point,
                    Margin = new Thickness(10, Array.IndexOf(bulletPoints, point) == 0 ? 5 : 0, 0, 0)
                });
            }

            // Assemble the components
            border.Content = contentLayout;

            // Add to the summary tab
            summaryTabContent.Add(border);
        }

        private async void OnExportMeetingClicked(object sender, EventArgs e)
        {
            try
            {
                // Create a meeting record from the current state
                var meetingRecord = new MeetingRecord
                {
                    Title = "Meeting " + MeetingStartTime.ToString("yyyy-MM-dd HH:mm"),
                    StartTime = MeetingStartTime,
                    Transcripts = Transcripts.ToList()
                };

                // Get summary data if available
                var summaryTabContent = (VerticalStackLayout)SummaryTab.Content;
                if (summaryTabContent.Children.Count > 0)
                {
                    var summary = new MeetingSummary();
                    summary.Title = "Meeting Summary";
                    summary.Points = [];

                    // Extract summary sections from the UI
                    foreach (var child in summaryTabContent.Children)
                    {
                        if (child is Border border && border.Content is VerticalStackLayout layout && layout.Children.Count > 0)
                        {
                            var titleLabel = layout.Children[0] as Label;
                            if (titleLabel != null)
                            {
                                var points = new List<string>();
                                // Extract bullet points (skip the title)
                                for (int i = 1; i < layout.Children.Count; i++)
                                {
                                    if (layout.Children[i] is Label pointLabel)
                                    {
                                        // Remove bullet character if present
                                        string text = pointLabel.Text;
                                        if (text.StartsWith("• "))
                                        {
                                            text = text.Substring(2);
                                        }
                                        points.Add(text);
                                    }
                                }
                                summary.Points.Add(new() { Title = titleLabel.Text, Contents = points.ToArray() });
                            }
                        }
                    }
                    meetingRecord.Summary = SummaryMarkdown.MarkdownText;
                }

                // Serialize to JSON
                var jsonOptions = new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true,
                    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                };
                var jsonContent = System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(meetingRecord, jsonOptions);

                // Get the filename for saving
                string defaultFilename = $"Meeting_{MeetingStartTime:yyyyMMdd_HHmmss}.json";

                // Save to file using FileSaver
                await FileSaver.Default.SaveAsync(defaultFilename, new MemoryStream(jsonContent));
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", $"Error exporting meeting records: {ex.Message}", "OK");
            }
        }

        private async void OnImportMeetingClicked(object sender, EventArgs e)
        {
            try
            {
                // Open file picker to select a JSON file
                var options = new PickOptions
                {
                    PickerTitle = "Select Meeting Records File",
                    FileTypes = new FilePickerFileType(
                        new Dictionary<DevicePlatform, IEnumerable<string>>
                        {
                    { DevicePlatform.iOS, new[] { "public.json" } },
                    { DevicePlatform.Android, new[] { "application/json" } },
                    { DevicePlatform.WinUI, new[] { ".json" } },
                    { DevicePlatform.MacCatalyst, new[] { "json" } }
                        })
                };

                var result = await FilePicker.Default.PickAsync(options);
                if (result == null)
                    return;

                // Read the JSON file
                var jsonContent = await File.ReadAllTextAsync(result.FullPath);

                // Deserialize to MeetingRecord
                var jsonOptions = new System.Text.Json.JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                };
                var meetingRecord = System.Text.Json.JsonSerializer.Deserialize<MeetingRecord>(jsonContent, jsonOptions);

                if (meetingRecord == null)
                {
                    await DisplayAlert("Error", "Unable to load meeting records, format may be incorrect", "OK");
                    return;
                }

                // Clear current UI state
                TranscriptArea.Clear();
                Transcripts.Clear();

                // Set meeting start time
                MeetingStartTime = meetingRecord.StartTime;

                // Load transcripts
                Transcripts.AddRange(meetingRecord.Transcripts);

                // Rebuild transcript UI
                foreach (var entry in Transcripts)
                {
                    AddTranscriptEntryToUI(entry);
                }

                // Load summary if available
                if (!string.IsNullOrEmpty(meetingRecord.Summary))
                {
                    SummaryMarkdown.MarkdownText = meetingRecord.Summary;
                }

                await DisplayAlert("Success", $"Meeting records imported: {meetingRecord.Title}", "OK");

                // Switch to transcript tab
                OnTranscriptTabClicked(this, EventArgs.Empty);
            }
            catch (Exception ex)
            {
                await DisplayAlert("Error", $"Error importing meeting records: {ex.Message}", "OK");
            }
        }

        public void DisableButton(Button b)
        {
            b.IsEnabled = false;
            b.BackgroundColor = Color.FromArgb("e2e8f0");
            b.TextColor = Colors.Gray;
        }
        public void EnableButton(Button b)
        {
            b.IsEnabled = true;
            b.BackgroundColor = Color.FromArgb("3b82f6");
            b.TextColor = Colors.White;
        }

        bool translateRecording = false;
        WaveFileWriter? transalteRecordingWriter = null;
        private async void OnTranslateRecording(object sender, EventArgs e)
        {
            if (translateRecording)
            {
                DisableButton(MicButton);
                translateRecording = false;
                transalteRecordingWriter!.Close();
                transalteRecordingWriter!.Dispose();
                using var fs = File.OpenRead("translate.wav");
                var request = new SpeechRecognitionRequest()
                {
                    AudioData = Google.Protobuf.ByteString.FromStream(fs),
                    ModelSize = "small",
                    Language = "auto",
                    ReturnTimestamps = false,
                };
                var result = await Client.SpeechRecognitionAsync(request);

                SourceTextEditor.Text = string.Join("\n", result.TranscribedText);

                EnableButton(MicButton);
                MicButton.Text = "🎤 Start Recording";
            }
            else
            {
                MicButton.Text = "🎤 Stop Recording";
                transalteRecordingWriter = new WaveFileWriter("translate.wav", Microphone.Format);
                translateRecording = true;
            }
        }

        public string ToLangCode(string lang)
        {
            return lang switch
            {
                "Chinese" or "中文" => "zh",
                "English" or "英文" => "en",
                "Japanese" or "日文" => "ja",
                "Korean" or "韓文" => "ko",
                "French" or "法文" => "fr",
                "German" or "德文" => "de",
                _ => "auto"
            };
        }

        private async void OnTranslateClicked(object sender, EventArgs e)
        {
            DisableButton(TranslateButton);
            if (SourceLangPicker.SelectedItem == null || TargetLangPicker.SelectedItem == null)
            {
                await DisplayAlert("Error", "Please select source and target languages", "OK");
                EnableButton(TranslateButton);
                return;
            }
            var translateResult = await Translator.TranslateAsync(new TranslateRequest()
            {
                SourceLanguage = ToLangCode(SourceLangPicker.SelectedItem.ToString()),
                TargetLanguage = ToLangCode(TargetLangPicker.SelectedItem.ToString()),
                TextToTranslate = SourceTextEditor.Text
            });
            if (translateResult != null)
            {
                for (int i = 0; i < translateResult.TranslatedText.Length; i++)
                {
                    TranslatedTextEditor.Text = translateResult.TranslatedText.Substring(0, i);
                    await Task.Delay(20);
                }
            }
            else
            {
                await DisplayAlert("Error", "Translation failed, please try again later", "OK");
            }
            EnableButton(TranslateButton);
        }

        byte[] imgData;
        private async void OnPlayTranslateAudio(object sender, EventArgs e)
        {
            DisableButton(SpeakButton);
            using var fs = File.OpenRead("D:\\tts_sample.mp3");
            var req = new TtsRequest() { Language = ToLangCode(TargetLangPicker.SelectedItem.ToString()), TextToSpeak = TranslatedTextEditor.Text, ReferenceAudio = ByteString.FromStream(fs) };
            var ttsResult = await Client.TtsAsync(req);
            using var ttsFs = File.Create("tts_result.wav");
            ttsResult.GeneratedAudio.WriteTo(ttsFs);
            ttsFs.Seek(0, SeekOrigin.Begin);
            var lipReq = new Wav2LipRequest() { AudioData = ByteString.FromStream(ttsFs), ImageData = ByteString.FromStream(new MemoryStream(imgData)) };
            var lipResult = await Client.Wav2LipAsync(lipReq);
            using var lipFs = File.Create("lip_result.mp4");
            lipResult.VideoData.WriteTo(lipFs);
            EnableButton(SpeakButton);
        }
    }

    public class MeetingRecord
    {
        public string Title { get; set; } = string.Empty;
        public DateTime StartTime { get; set; }
        public string Summary { get; set; } = "";
        public List<TranscriptEntry> Transcripts { get; set; } = [];
    }
    public class MeetingSummary
    {
        public string Title { get; set; } = string.Empty;
        public List<BulletPoint> Points { get; set; } = [];
    }

    public class BulletPoint
    {
        public string Title { get; set; } = string.Empty;
        public string[] Contents { get; set; } = [];
    }
}
