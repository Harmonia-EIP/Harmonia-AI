# Change Log

**2025** : First version of AudioLDM.

## Command Usage
Prepare running environment
```shell
# Install AudioLDM
pip3 install -e .
# Running a text-to-audio generation test
audioldm --text "A hammer is hitting a wooden surface"
```


**Audio-to-Audio Generation**: generate an audio guided by an audio (output will have similar audio events as the input audio file).
```shell
audioldm --file_path trumpet.wav
# Result will be saved in "./output/generation_audio_to_audio/trumpet"
```

**Text-guided Audio-to-Audio Style Transfer**
```shell
# Test run
# --file_path is the original audio file for transfer
# -t is the text AudioLDM uses for transfer. 
# Please make sure that --file_path exist
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing" 
# Result will be saved in "./output/transfer/trumpet"

# Tune the value of --transfer_strength is important!
# --transfer_strength: A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing" --transfer_strength 0.25
```

 How to choose between different model checkpoints?
```
# Add the --model_name parameter, choice={audioldm-m-text-ft, audioldm-s-text-ft, audioldm-m-full, audioldm-s-full,audioldm-l-full,audioldm-s-full-v2}
audioldm --model_name audioldm-s-full
```


 For more options on guidance scale, batchsize, seed, ddim steps, etc., please run
```shell
audioldm -h
```
```console
usage: audioldm [-h] [--mode {generation,transfer}] [-t TEXT] [-f FILE_PATH] [--transfer_strength TRANSFER_STRENGTH] [-s SAVE_PATH] [--model_name {audioldm-s-full,audioldm-l-full,audioldm-s-full-v2}] [-ckpt CKPT_PATH]
                [-b BATCHSIZE] [--ddim_steps DDIM_STEPS] [-gs GUIDANCE_SCALE] [-dur DURATION] [-n N_CANDIDATE_GEN_PER_TEXT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --mode {generation,transfer}
                        generation: text-to-audio generation; transfer: style transfer
  -t TEXT, --text TEXT  Text prompt to the model for audio generation, DEFAULT ""
  -f FILE_PATH, --file_path FILE_PATH
                        (--mode transfer): Original audio file for style transfer; Or (--mode generation): the guidance audio file for generating simialr audio, DEFAULT None
  --transfer_strength TRANSFER_STRENGTH
                        A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text, DEFAULT 0.5
  -s SAVE_PATH, --save_path SAVE_PATH
                        The path to save model output, DEFAULT "./output"
  --model_name {audioldm-s-full,audioldm-l-full,audioldm-s-full-v2}
                        The checkpoint you gonna use, DEFAULT "audioldm-s-full"
  -ckpt CKPT_PATH, --ckpt_path CKPT_PATH
                        (deprecated) The path to the pretrained .ckpt model, DEFAULT None
  -b BATCHSIZE, --batchsize BATCHSIZE
                        Generate how many samples at the same time, DEFAULT 1
  --ddim_steps DDIM_STEPS
                        The sampling step for DDIM, DEFAULT 200
  -gs GUIDANCE_SCALE, --guidance_scale GUIDANCE_SCALE
                        Guidance scale (Large => better quality and relavancy to text; Small => better diversity), DEFAULT 2.5
  -dur DURATION, --duration DURATION
                        The duration of the samples, DEFAULT 10
  -n N_CANDIDATE_GEN_PER_TEXT, --n_candidate_gen_per_text N_CANDIDATE_GEN_PER_TEXT
                        Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation, DEFAULT 3
  --seed SEED           Change this value (any integer number) will lead to a different generation result. DEFAULT 42
```

# Hugging Face Diffusers

AudioLDM is available in the Hugging Face [🧨 Diffusers](https://github.com/huggingface/diffusers) library from v0.15.0 onwards. The official checkpoints can be found on the [Hugging Face Hub](https://huggingface.co/cvssp), alongside [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm) and [examples scripts](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm).

To install Diffusers and Transformers, run:
```bash
pip install --upgrade diffusers transformers
```

You can then load pre-trained weights into the [AudioLDM pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm) and generate text-conditional audio outputs:
```python
from diffusers import AudioLDMPipeline
import torch

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
```
