using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Numpy;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace IMMA
{
    public class TtsOnnxVocoder
    {
        private readonly InferenceSession _session;
        private readonly string _onnxModelPath;
        private readonly string _melSpectrogramPath;
        private readonly string _speakerEmbeddingPath;

        public TtsOnnxVocoder(string onnxModelPath, string melSpectrogramPath, string speakerEmbeddingPath)
        {
            _onnxModelPath = onnxModelPath;
            _melSpectrogramPath = melSpectrogramPath;
            _speakerEmbeddingPath = speakerEmbeddingPath;

            // Load the ONNX model
            // For MAUI, assets are embedded resources. We need to copy them to a temporary file.
            var modelStream = FileSystem.Current.OpenAppPackageFileAsync(_onnxModelPath).Result;
            var tempModelPath = Path.Combine(FileSystem.Current.CacheDirectory, Path.GetFileName(_onnxModelPath));
            using (var fileStream = new FileStream(tempModelPath, FileMode.Create, FileAccess.Write))
            {
                modelStream.CopyTo(fileStream);
            }
            _session = new InferenceSession(tempModelPath);
        }

        public async Task<float[]> GenerateAudioAsync()
        {
            // Load mel spectrogram and speaker embedding from .npy files
            // These are also embedded resources, so copy to temp files first.
            var melStream = await FileSystem.Current.OpenAppPackageFileAsync(_melSpectrogramPath);
            var tempMelPath = Path.Combine(FileSystem.Current.CacheDirectory, Path.GetFileName(_melSpectrogramPath));
            using (var fileStream = new FileStream(tempMelPath, FileMode.Create, FileAccess.Write))
            {
                melStream.CopyTo(fileStream);
            }
            var melSpectrogram = np.Load(tempMelPath);

            var speakerStream = await FileSystem.Current.OpenAppPackageFileAsync(_speakerEmbeddingPath);
            var tempSpeakerPath = Path.Combine(FileSystem.Current.CacheDirectory, Path.GetFileName(_speakerEmbeddingPath));
            using (var fileStream = new FileStream(tempSpeakerPath, FileMode.Create, FileAccess.Write))
            {
                speakerStream.CopyTo(fileStream);
            }
            var speakerEmbedding = np.Load(tempSpeakerPath);

            // Prepare inputs for ONNX Runtime
            // ONNX Runtime expects float arrays, so convert from Numpy.NDarray
            var melTensor = new DenseTensor<float>(melSpectrogram.ToArray<float>(), melSpectrogram.shape.Dimensions.Select(d => (long)d).ToArray());
            var speakerTensor = new DenseTensor<float>(speakerEmbedding.ToArray<float>(), speakerEmbedding.shape.Dimensions.Select(d => (long)d).ToArray());

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("mel_spectrogram", melTensor),
                NamedOnnxValue.CreateFromTensor("speaker_embedding", speakerTensor)
            };

            // Run inference
            using (var results = _session.Run(inputs))
            {
                var audioOutput = results.FirstOrDefault(item => item.Name == "audio_output"); // Assuming output name is "audio_output"
                if (audioOutput != null)
                {
                    return audioOutput.AsTensor<float>().ToArray();
                }
                else
                {
                    throw new System.Exception("ONNX inference did not return 'audio_output' tensor.");
                }
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}