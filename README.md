# About this project
The project is to show how to run Whisper on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

The source code is based on [RyzenAI-SW Whisper Demo](https://github.com/amd/RyzenAI-SW/tree/419829fc8f8f58ad1a31c4fcc0287d2103f84824/demo/ASR/Whisper)

# Quick Steps
## Prepare model
### Install required packages
```
# make sure to use Python 3.11, 3.12 and later versions will fail in compiling onnxsim
python --version
pip install -r requirements.txt
```
### Export model to onnx
```
optimum-cli export onnx --model openai/whisper-base --opset 18 exported_whisper_base
```
### Convert model (from dynamic to static)
```
python dynamic_to_static.py --input_model_dir exported_whisper_base
```
## Run
### Install and setup OpenVINO
```
curl -o ov_2025_3.zip https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.3/windows/openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64.zip
tar -zxf ov_2025_3.zip
.\openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64\setupvars.bat
```
### Install FFmpeg
```FFmpeg``` is required as ```torchcodec``` leverages ```FFmpeg``` as its underlying encoding/decoding engine.

1. Download [```ffmpeg-7.1.1-full_build-shared.zip```](https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build-shared.zip) from [```ffmpeg releases repo```](https://github.com/GyanD/codexffmpeg/releases). Don’t use release 8.x as ```torchcodec``` supports ```FFmpeg``` 4.x~7.x.
2. Input "```pip show pip```" to find your Python site-packages location.
3. Decompress the downloaded ```FFmpeg``` packagein step 1., copy ```bin\*.dll``` to Python ```site-packages\torchcodec```. The files under ```site-packages\torchcodec``` should look like

```
C:\Python\openvino_env\Lib\site-packages\torchcodec>dir /o
 Volume in drive C is OSDisk
 Volume Serial Number is C2C8-D7B9

 Directory of C:\Python\openvino_env\Lib\site-packages\torchcodec

09/11/2025  09:18 AM    <DIR>          .
09/10/2025  05:26 PM    <DIR>          ..
09/09/2025  05:32 PM    <DIR>          __pycache__
09/09/2025  05:32 PM    <DIR>          _core
09/09/2025  05:32 PM    <DIR>          _samplers
09/09/2025  05:32 PM    <DIR>          decoders
09/09/2025  05:32 PM    <DIR>          encoders
09/09/2025  05:32 PM    <DIR>          samplers
09/09/2025  05:32 PM               595 __init__.py
09/09/2025  05:32 PM             5,350 _frame.py
09/09/2025  05:32 PM             2,422 _internally_replaced_utils.py
03/12/2025  02:06 PM        89,117,184 avcodec-61.dll
03/12/2025  02:06 PM         4,514,816 avdevice-61.dll
03/12/2025  02:06 PM        41,882,624 avfilter-10.dll
03/12/2025  02:06 PM        18,723,328 avformat-61.dll
03/12/2025  02:06 PM         2,840,576 avutil-59.dll
09/09/2025  05:32 PM           310,784 libtorchcodec_core4.dll
09/09/2025  05:32 PM           310,784 libtorchcodec_core5.dll
09/09/2025  05:32 PM           310,784 libtorchcodec_core6.dll
09/09/2025  05:32 PM           310,784 libtorchcodec_core7.dll
09/09/2025  05:32 PM           564,736 libtorchcodec_custom_ops4.dll
09/09/2025  05:32 PM           564,736 libtorchcodec_custom_ops5.dll
09/09/2025  05:32 PM           564,736 libtorchcodec_custom_ops6.dll
09/09/2025  05:32 PM           564,736 libtorchcodec_custom_ops7.dll
09/09/2025  05:32 PM           204,288 libtorchcodec_pybind_ops4.pyd
09/09/2025  05:32 PM           204,288 libtorchcodec_pybind_ops5.pyd
09/09/2025  05:32 PM           204,288 libtorchcodec_pybind_ops6.pyd
09/09/2025  05:32 PM           204,288 libtorchcodec_pybind_ops7.pyd
03/12/2025  02:06 PM            87,552 postproc-58.dll
03/12/2025  02:06 PM           438,784 swresample-5.dll
03/12/2025  02:06 PM           707,584 swscale-8.dll
09/09/2025  05:32 PM                75 version.py
              24 File(s)    162,640,122 bytes
               8 Dir(s)  115,870,347,264 bytes free
```

### Install OpenVINO execution provider
```
pip install onnxruntime-openvino
```

### Run the pipeline (input from a file)
```
python run_whisper.py --model-dir exported_whisper_base --device cpu --input audio_files/61-52s.wav
```
* The device can be ```cpu```, ```gpu```, ```npu``` or ```ov_cpu```
* :warning: Don't use ```venv``` to run the pipeline. Somehow ```openvino*.dll``` can not be found under ```venv``` virtual environments

### Run the pipeline (input from microphone)
```
python run_whisper.py --model-dir exported_whisper_base --device cpu --input mic
```
### Run the pipeline to evaluate a dataset
```
python run_whisper.py --model-dir exported_whisper_base --device cpu --eval-dir eval_dataset\LibriSpeech-samples
```
* Results will be stored in ```results\LibriSpeech-samples\results.txt```
## Log (NPU)
```
C:\Github\whisper-ovep-python>python run_whisper.py --model-dir exported_whisper_base --device npu --input audio_files/61-52s.wav
Selected provider: ['OpenVINOExecutionProvider']
Provider option: [{'device_type': 'NPU', 'cache_dir': './cache'}]

Performance Metric (Chunk 1):
 Time to First Token for this chunk: 0.11 seconds

Performance Metric (Chunk 2):
 Time to First Token for this chunk: 2.17 seconds
 RTF: 0.08

Transcription: Also, there was a stripling page who turned into a maze was so sweet a lady, sir, and in some manner I do think she died. But then the picture was gone as quickly as it came. Sister Nell, do you hear these marvels? Take your place and let us see what the crystal can show to you, like is not young master, though I am an old man. With all rant the opening of the tent to see what might be a miss, but Master Will, who peeped out first, needed no more than one glance. Mistress Fitzuth to the rear of the tent cries of unnotting him, unnotting him. Before them fled the stroller and his three sons, capless and tear what is that tumult and rioting, cried out the squire, thoratatively, and he blew twice on the silver whistle which hung at his belt.

C:\Github\whisper-ovep-python>
```
[Full log](https://github.com/luke-lin-vmc/whisper-ovep-python/blob/main/log_full.txt) (from scratch) is provided for reference

# Reference
https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html


# Original [RyzenAI-SW](https://github.com/amd/RyzenAI-SW) README.md
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

