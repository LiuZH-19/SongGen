# SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation

🚀🚀🚀 Official implementation of **SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation**
<p align="center" style="font-size: 1 em; margin-top: -1em">
<a href="https://scholar.google.com/citations?user=iELd-Q0AAAAJ">Zihan Liu</a>,  
<a href="https://mark12ding.github.io/">Shuangrui Ding</a>,  
<a href="https://github.com/rookiexiong7/">Zhixiong Zhang</a>, 
<a href="https://lightdxy.github.io/">Xiaoyi Dong</a>,  
<a href="https://panzhang0212.github.io/">Pan Zhang</a>,
<a href="https://yuhangzang.github.io/">Yuhang Zang</a>,  
<a href="https://scholar.google.com/citations?user=sJkqsqkAAAAJ">Yuhang Cao</a>, </br>  
<a href="http://dahua.site/">Dahua Lin</a>,  
<a href="https://myownskyw7.github.io/">Jiaqi Wang</a> 
</p>

<p align="center" style="font-size: 5 em; margin-top: 0.5em">
<a href="https://arxiv.org/abs/2502.13128"><img src="https://img.shields.io/badge/arXiv-<color>"></a>
<a href="https://github.com/LiuZH-19/SongGen"><img src="https://img.shields.io/badge/Code-red"></a>
<a href="https://liuzh-19.github.io/SongGen/"><img src="https://img.shields.io/badge/Demo-20d67c"></a>
<a href="https://huggingface.co/collections/LiuZH-19/songgen-a-single-stage-auto-regressive-transformer-for-text-6867ec21169d808034f6d252">
    <img src="https://img.shields.io/badge/HF-Collection-yellow"></a>
</p>





## 📜 News
🚀 [2025/7/4] We released the training code along with a detailed [training guide](./training/README.md) .

🚀 [2025/6/30] The MusicCaps Test Set is now available on [Huggingface🤗](https://huggingface.co/datasets/LiuZH-19/MusicCaps_Test_Song) for text-to-song eveluation.

🚀 [2025/6/27] We released the checkpoint of SongGen Interleaving (A-V) at [Huggingface🤗](https://huggingface.co/LiuZH-19/SongGen_interleaving_A_V).

🎉 [2025/5/1] SongGen is accepted by ICML 2025!

🚀 [2025/3/18] We released the checkpoint of SongGen Mixed_Pro at [Huggingface🤗](https://huggingface.co/LiuZH-19/SongGen_mixed_pro).

🚀 [2025/2/19] The [paper](https://arxiv.org/abs/2502.13128) and [demo page](https://liuzh-19.github.io/SongGen/) are released!

## 💡 Highlights
- 🔥We introduce SongGen, a **single-stage** auto-regressive transformer for **text-to-song** generation, offering versatile control via lyrics, descriptive text, and an optional reference voice.
- 🔥SongGen supports both **mixed** and **dual-track mode** to accommodate diverse requirements. Our experiments provide **valuable insights** for optimizing both modes.
- 🔥By releasing the **model weights**, **code**, **annotated data**, and **preprocessing pipeline**, we aim to establish a simple yet effective baseline for future song generation research.
<!-- <img align="center" src="assets/imgs/motivation1.jpg" style="  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;" /> -->

## 👨‍💻 Todo
- [ ] Release annotated data and preprocessing pipeline
- [x] Release Musiccaps Test set
- [x] Release SongGen training code
- [x] Release SongGen (Interleaving A-V) checkpoint
- [x] Release SongGen Mixed_pro checkpoint
- [x] Release SongGen inference code 
- [x] SongGen demo

## 🛠️ Usage

### 1. Install environment and dependencies
```bash
git clone https://github.com/LiuZH-19/SongGen.git
cd SongGen
# We recommend using conda to create a new environment.
conda create -n songgen_env python=3.9.18 
conda activate songgen_env
# Install CUDA >= 11.8 and PyTorch, e.g.,
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.6.1 --no-build-isolation
```
To use SongGen only in inference mode, install it using:
```bash
pip install -e .
```
### 2. Download the xcodec

Download the X-Codec checkpoint from [🤗](
https://huggingface.co/ZhenYe234/xcodec/blob/main/xcodec_hubert_general_audio_v2.pth) and place it in the following directory : SongGen/songgen/xcodec_wrapper/xcodec_infer/ckpts/general_more

```
xcodec_infer
    ├── ckpts
    │   └── general_more
    │       ├── config_hubert_general.yaml
    │       └── xcodec_hubert_general_audio_v2.pth

```

### 3. Run the inference

#### (1). Mixed Pro Mode

```python
import torch
import os
from songgen import (
    VoiceBpeTokenizer,
    SongGenMixedForConditionalGeneration,
    SongGenProcessor
)
import soundfile as sf

ckpt_path = "LiuZH-19/SongGen_mixed_pro" # Path to the pretrained model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SongGenMixedForConditionalGeneration.from_pretrained(
    ckpt_path,
    attn_implementation='sdpa').to(device)
processor = SongGenProcessor(ckpt_path, device)

# Define input text and lyrics
lyrics = "..." # The lyrics text
text = "..." # The music description text
ref_voice_path = 'path/to/your/reference_audio.wav' # Path to reference audio, optional
separate= True # Whether to separate the vocal track from the reference voice audio

model_inputs = processor(text=text, lyrics=lyrics, ref_voice_path=ref_voice_path, separate=separate) 
generation = model.generate(**model_inputs,
                do_sample=True,
            )
audio_arr = generation.cpu().numpy().squeeze()
sf.write("songgen_out.wav", audio_arr, model.config.sampling_rate)
```



#### (2). Interleaving A-V  (Dual-track mode)
```python
import torch
import os
from songgen import (
    VoiceBpeTokenizer,
    SongGenDualTrackForConditionalGeneration,
    SongGenProcessor
)
import soundfile as sf

ckpt_path = "LiuZH-19/SongGen_interleaving_A_V" # Path to the pretrained model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SongGenDualTrackForConditionalGeneration.from_pretrained(
    ckpt_path,
    attn_implementation='sdpa').to(device)
processor = SongGenProcessor(ckpt_path, device)

# Define input text and lyrics
lyrics = "..." # The lyrics text
text = "..." # The music description text
ref_voice_path = 'path/to/your/reference_audio.wav' # Path to reference audio, optional
separate= True # Whether to separate the vocal track from the reference voice audio

model_inputs = processor(text=text, lyrics=lyrics, ref_voice_path=ref_voice_path, separate=True) 
generation = model.generate(**model_inputs,
                do_sample=True,
            )

acc_array = generation[0].cpu().numpy()
vocal_array = generation[1].cpu().numpy()
min_len =min(vocal_array.shape[0], acc_array.shape[0])
acc_array = acc_array[:min_len]
vocal_array = vocal_array[:min_len]
audio_arr = vocal_array + acc_array
sf.write("songgen_out.wav", audio_arr, model.config.sampling_rate)
```

### 4. Training

The [training folder](./training) contains all the information to train or fine-tune your own SongGen model. See the [training guide](./training/README.md) for step-by-step instructions.



## ❤️ Acknowledgments
This library builds on top of a number of open-source giants, to whom we'd like to extend our warmest thanks for providing these tools!

Special thanks to:

- [Parler-tts](https://github.com/huggingface/parler-tts): The codebase we built upon. 
- [X-Codec](https://github.com/zhenye234/xcodec): The audio codec utilized in our research.
- [lp-music-caps](https://github.com/seungheondoh/lp-music-caps): A project aimed at generating captions for music. 

We deeply appreciate all the support we've received along the way.

## ☎️ Limitation and Future Work

This is a **research work** focused on **text-to-song** generation. Due to the limitations of the current training dataset, our model is currently restricted to generating English songs with a maximum duration of 30 seconds.
However, despite being trained on only **2k hours** of data with a **1.3B** parameter model, our approach has demonstrated strong effectiveness and promising potential in generating coherent and expressive songs. We believe that scaling up both data and model size will further enhance lyrics alignment and musicality.
That being said, scaling the dataset is time-consuming and challenging. We welcome collaborations and discussions to explore new ways to improve the model and extend its capabilities.
For any inquiries or potential collaborations, feel free to reach out: Zihan Liu (liuzihan@pjlab.org.cn) and Jiaqi Wang (wangjiaqi@pjlab.org.cn).

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{liu2025songgen,
      title={SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation}, 
      author={Zihan Liu and Shuangrui Ding and Zhixiong Zhang and Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Dahua Lin and Jiaqi Wang},
      year={2025},
      eprint={2502.13128},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.13128}, 
}

```





