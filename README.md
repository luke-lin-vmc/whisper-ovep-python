# Quick Steps
## Export Model
```
# make sure to ues Python 3.11, 3.12 and 3.13 will fail in compiling onnxsim
pip install -r requirements.txt
optimum-cli export onnx --model openai/whisper-base.en --opset 17 exported_whisper_base
python dynamic_to_static.py --input_model_dir exported_whisper_base
```
## Run
```
curl -o ov_2025_1.zip https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.1/windows/openvino_toolkit_windows_2025.1.0.18503.6fec06580ab_x86_64.zip
tar -zxf ov_2025_1.zip
.\openvino_toolkit_windows_2025.1.0.18503.6fec06580ab_x86_64\setupvars.bat
pip install onnxruntime-openvino
python run_whisper.py --encoder exported_whisper_base/encoder_model_static.onnx --decoder exported_whisper_base/decoder_model_static.onnx --model-type whisper-base --config-file config/model_config.json --device cpu --input audio_files/61-52s.wav
```

# Log
```
(ovep_venv) C:\GitHub\whisper-ovep-python>python run_whisper.py --encoder exported_model_directory/encoder_model_static.onnx --decoder exported_model_directory/decoder_model_static.onnx --model-type whisper-base --config-file config/model_config.json --device cpu --input audio_files/61-52s.wav
Selected Provider Options:
Decoder:  ['CPUExecutionProvider']
Encoder:  ['CPUExecutionProvider']

Loading tokenizer and feature extractor from: C:\GitHub\whisper-ovep-python\exported_model_directory
C:\Python\ovep_venv\Lib\site-packages\torchaudio\_backend\utils.py:213: UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.
  warnings.warn(

Performance Metric (Chunk 1):
 Time to First Token for this chunk: 1.87 seconds

Performance Metric (Chunk 2):
 Time to First Token for this chunk: 44.99 seconds
 RTF: 1.51

Transcription: Also, there was a stripling page who turned into a maze with so sweet a lady, sir. And in some manner I do think she died. But then the picture was gone as quickly as it came. Sister Nell, do you hear these mottles? Take your place and let us see what the crystal can show to you, like his not young master. Though I am an old man. With all rant the opening of the tent to see what might be a miss. But Master Will, who peeped out first, needed no more than one glance. Mistress Fitzsuth to the rear of the Ted cries of "A knotting ham! A knotting ham!" before them fled the stroller and his three sons, "Capless and terrible!" "What is that tumult and rioting?" cried out the squire, authoritatively, and he blew twice on the silver whistle which hung at his belt.
```
# Reference
https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html

# Original README.md
---
<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Automatic Speech Recognition </h1>
    </td>
 </tr>
</table>

# Automatic Speech Recognition using OpenAI Whisper

Unlock fast, on-device speech recognition with RyzenAI and OpenAI’s Whisper. This demo walks you through preparing and running OpenAI's Whisper (base, small, medium) for fast, local ASR on AMD NPU.

## Features

* 🚀 Export Whisper models from Hugging Face to ONNX
* ⚙️ Optimize for static shape inference
* ⚡ Run ASR locally on CPU or NPU
* 📊 Evaluate ASR on LibriSpeech samples and report WER/CER
* 🎧 Supports transcription of audio files and microphone input
* ⏱️ Reports Performance using RTF and TTFT

## 🔗 Quick Links
- [Prerequisites](#prerequisites)
- [Export Whisper Model to ONNX](#export-whisper-model-to-onnx)
- [Accelerate Whisper on AMD NPU](#accelerate-whisper-on-amd-npu)
  - [Why run on NPU?](#why-run-on-npu)
  - [Set up VitisEP Configuration for NPU](#set-up-vitisep-configuration-for-npu)
- [ Usage](#usage)
  - [Transcribe Audio File](#transcribe-audio-file)
  - [Transcribe from Microphone](#transcribe-from-microphone)
  - [Evaluate on Dataset](#evaluate-on-dataset)
- [ Notes](#notes)

## 📦 Prerequisites

1. **Install Ryzen AI SDK**
   Follow [RyzenAI documentation](https://ryzenai.docs.amd.com/en/latest/inst.html#) to install SDK and drivers.

2. **Activate environment**

   ```bash
   conda activate ryzen-ai-1.5.0
   ```

3. **Clone repository**

   ```bash
   git clone https://github.com/amd/RyzenAI-SW.git
   cd RyzenAI-SW/example/ASR/Whisper-AI
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```


## ⚙️ Export Whisper Model to ONNX

1. **Export using Hugging Face Optimum**

   ```bash
   optimum-cli export onnx --model openai/whisper-base.en --opset 17 exported_model_directory
   ```

   * Output: `encoder_model.onnx`, `decoder_model.onnx`
   * Supports multilingual whisper-base, whisper-small, whisper-medium

2. **Convert dynamic ONNX to static**

   ```bash
   python dynamic_to_static.py --input_model exported_model_directory/encoder_model.onnx
   python dynamic_to_static.py --input_model exported_model_directory/decoder_model.onnx
   ```

   * Uses `onnxruntime.tools.make_dynamic_shape_fixed`
   * Final models overwrite originals in `exported_model_directory`

## ⚡Accelerate Whisper on AMD NPU

### Why run on NPU?

* Offloads compute from CPU onto NPU, freeing up CPU for other tasks.
* Delivers higher throughput and lower power consumption when running AI workloads
* Optimized execution of Whisper’s encoder and decoder models.
* Runs models with BFP16 precision for near-FP32 accuracy and INT8-like performance.

#### NPU Run for Whisper-Base
When running inference on the NPU, 100% of the encoder operators and 93.4% of the decoder operators are executed on the NPU.
```bash
   #encoder operations
   [Vitis AI EP] No. of Operators : VAIML   225
   [Vitis AI EP] No. of Subgraphs : VAIML     1

   #decoder operations
   [Vitis AI EP] No. of Operators :   CPU    24  VAIML   341
   [Vitis AI EP] No. of Subgraphs : VAIML     2
```
#### Set up VitisEP Configuration for NPU

* Edit `config/model_config.json` to specify Execution Providers.
* For NPU:

  * Set `cache_key` and `cache_dir`
  * Use corresponding `vitisai_config` from `config/`

Example:

```json
{
  "config_file": "config/whisper_vitisai.json",
  "cache_dir": "./cache",
  "cache_key": "whisper_base"
}
```
#### ⚠️ Special Instructions for Whisper-Medium
When running whisper-medium on NPU, it is recommended to add the following flags to `configs\vitisai_config_whisper_encoder.json` incase of compilation issues.
```json
"vaiml_config": {
  "optimize_level": 2,
  "aiecompiler_args": "--system-stack-size=512"
}
```
These settings:

- optimize_level=2: Enables aggressive optimizations for larger models.
- --system-stack-size=512: Increases the AI Engine system stack size to handle Whisper-Medium’s higher resource demand.

## 🚀 Usage

### Transcribe Audio File
Use this to transcribe a pre-recorded `.wav` file into text using the Whisper mode
```bash
python run_whisper.py \
  --encoder exported_model_directory/encoder_model.onnx \
  --decoder exported_model_directory/decoder_model.onnx \
  --model-type <whisper-type> \
  --config-file config/model_config.json \
  --device npu \
  --input path/to/audio.wav
```
- Replace <whisper-type> with whisper-base, whisper-small, or whisper-medium.

- Replace path/to/audio.wav with your audio file.

### Transcribe from Microphone
Run real-time speech-to-text by capturing audio from your microphone. This allows you to speak and see live transcription:

```bash
python run_whisper.py \
  --encoder exported_model_directory/encoder_model.onnx \
  --decoder exported_model_directory/decoder_model.onnx \
  --model-type <whisper-type> \
  --config-file config/model_config.json \
  --device npu \
  --input mic \
  --duration 0
```
- --duration 0 means continuous recording until stopped (Ctrl+C) or detects silence for a set duration

- Ideal for demos and testing live ASR performance.

### Evaluate on Dataset
Run batch evaluation on a dataset (e.g., LibriSpeech samples) to measure model performance with metrics like WER, CER, and RTF:
```bash
python run_whisper.py \
  --encoder exported_model_directory/encoder_model.onnx \
  --decoder exported_model_directory/decoder_model.onnx \
  --model-type <whisper-type> \
  --config-file config/model_config.json \
  --device npu \
  --eval-dir eval_dataset/LibriSpeech-samples \
  --results-dir results
```
- --eval-dir specifies the dataset directory.

- --results-dir is where evaluation reports (WER, CER, TTFT, RTF) will be saved.

- Useful for benchmarking and validating models.

## Notes

* First run on NPU may take \~15 min for model compilation.
* Ensure paths for encoder, decoder, and config files are correct.
* Supports CPU and NPU devices.

