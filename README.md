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
* Supported models: ```whisper-base```, ```whisper-small``` and ```whisper-medium```
### Convert model (from dynamic to static)
```
python dynamic_to_static.py --input_model_dir exported_whisper_base
```
## Run
### Install FFmpeg
```FFmpeg``` is required as ```torchcodec``` leverages ```FFmpeg``` as its underlying encoding/decoding engine.

1. `torchCodec` supports all major `FFmpeg` versions from 4.x~8.x. Here we download [```ffmpeg-8.1.2-full_build-shared.zip```](https://github.com/GyanD/codexffmpeg/releases/download/8.1.2/ffmpeg-8.1.2-full_build-shared.zip) from [```ffmpeg releases repo```](https://github.com/GyanD/codexffmpeg/releases)
2. Input ```pip show pip``` to find your Python site-packages location.
3. Decompress the downloaded ```FFmpeg``` package in step 1., copy ```bin\*.dll``` to Python ```site-packages\torchcodec```. The files under ```site-packages\torchcodec``` should look like

```
C:\Python\python313_venv\Lib\site-packages\torchcodec>dir /o
 Volume in drive C has no label.
 Volume Serial Number is 6E19-CE53

 Directory of C:\Python\python313_venv\Lib\site-packages\torchcodec

08/11/2026  11:22 AM    <DIR>          .
08/11/2026  11:11 AM    <DIR>          ..
08/11/2026  11:09 AM    <DIR>          __pycache__
08/11/2026  11:09 AM    <DIR>          _core
08/11/2026  11:09 AM    <DIR>          decoders
08/11/2026  11:09 AM    <DIR>          encoders
08/11/2026  11:09 AM    <DIR>          samplers
08/11/2026  11:09 AM    <DIR>          share
08/11/2026  11:09 AM    <DIR>          transforms
08/11/2026  11:09 AM             2,222 __init__.py
08/11/2026  11:09 AM             5,373 _frame.py
08/11/2026  11:09 AM             5,626 _internally_replaced_utils.py
08/11/2026  11:09 AM             1,722 _logging.py
06/27/2026  08:30 PM        97,454,080 avcodec-62.dll
06/27/2026  08:30 PM         6,323,200 avdevice-62.dll
06/27/2026  08:30 PM       124,344,320 avfilter-11.dll
06/27/2026  08:30 PM        20,179,968 avformat-62.dll
06/27/2026  08:30 PM         3,148,288 avutil-60.dll
08/11/2026  11:09 AM           920,576 libtorchcodec_core4.dll
08/11/2026  11:09 AM           921,088 libtorchcodec_core5.dll
08/11/2026  11:09 AM           921,088 libtorchcodec_core6.dll
08/11/2026  11:09 AM           921,088 libtorchcodec_core7.dll
08/11/2026  11:09 AM           922,624 libtorchcodec_core8.dll
08/11/2026  11:09 AM         1,607,168 libtorchcodec_custom_ops4.dll
08/11/2026  11:09 AM         1,607,168 libtorchcodec_custom_ops5.dll
08/11/2026  11:09 AM         1,607,168 libtorchcodec_custom_ops6.dll
08/11/2026  11:09 AM         1,607,168 libtorchcodec_custom_ops7.dll
08/11/2026  11:09 AM         1,607,168 libtorchcodec_custom_ops8.dll
08/11/2026  11:09 AM           152,064 libtorchcodec_pybind_ops4.pyd
08/11/2026  11:09 AM           152,064 libtorchcodec_pybind_ops5.pyd
08/11/2026  11:09 AM           152,064 libtorchcodec_pybind_ops6.pyd
08/11/2026  11:09 AM           152,064 libtorchcodec_pybind_ops7.pyd
08/11/2026  11:09 AM           152,064 libtorchcodec_pybind_ops8.pyd
08/11/2026  11:09 AM                 0 py.typed
06/27/2026  08:30 PM           486,912 swresample-6.dll
06/27/2026  08:30 PM        12,748,288 swscale-9.dll
08/11/2026  11:09 AM                80 version.py
              28 File(s)    278,102,703 bytes
               9 Dir(s)  454,825,242,624 bytes free
```
### Run the pipeline (input from a file)
```
python run_whisper.py --model-dir exported_whisper_base --device gpu --input audio_files/61-52s.wav
```
* The device can be
  * ```ov_cpu```, ```gpu``` or ```npu``` --- using CPU, GPU or NPU thru OpenVINOExecutionProvider
  * ```cpu``` --- using CPU thru default CPUExecutionProvider

### Run the pipeline (input from microphone)
```
python run_whisper.py --model-dir exported_whisper_base --device gpu --input mic
```
### Run the pipeline to evaluate a dataset
```
python run_whisper.py --model-dir exported_whisper_base --device gpu --eval-dir eval_dataset\LibriSpeech-samples
```
* Results will be stored in ```results\LibriSpeech-samples\results.txt```
## Sample Log
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
