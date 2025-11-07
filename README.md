# About this project
The project is to show how to run Whisper on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

The source code is forked from [RyzenAI-SW Whisper Demo](https://github.com/amd/RyzenAI-SW/tree/419829fc8f8f58ad1a31c4fcc0287d2103f84824/demo/ASR/Whisper)

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
### Install FFmpeg
```FFmpeg``` is required as ```torchcodec``` leverages ```FFmpeg``` as its underlying encoding/decoding engine.

1. Download [```ffmpeg-7.1.1-full_build-shared.zip```](https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-full_build-shared.zip) from [```ffmpeg releases repo```](https://github.com/GyanD/codexffmpeg/releases). Don’t use release 8.x as ```torchcodec``` supports ```FFmpeg``` 4.x~7.x.
2. Input ```pip show pip``` to find your Python site-packages location.
3. Decompress the downloaded ```FFmpeg``` package in step 1., copy ```bin\*.dll``` to Python ```site-packages\torchcodec```. The files under ```site-packages\torchcodec``` should look like

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
### Run the pipeline (input from a file)
```
python run_whisper.py --model-dir exported_whisper_base --device cpu --input audio_files/61-52s.wav
```
* The device can be ```cpu```, ```gpu```, ```npu``` or ```ov_cpu```
### Run the pipeline (input from microphone)
```
python run_whisper.py --model-dir exported_whisper_base --device cpu --input mic
```
### Run the pipeline to evaluate a dataset
```
python run_whisper.py --model-dir exported_whisper_base --device cpu --eval-dir eval_dataset\LibriSpeech-samples
```
* Results will be stored in ```results\LibriSpeech-samples\results.txt```
## Known issues
The following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
Solution is to simply reinstall ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino
```
## Log (device is NPU)
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
