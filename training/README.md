# Training SongGen

## 1. Getting started

To get started, you need to follow a few steps:
1. Install the requirements.
2. Find or initialize the model you'll train on. 
3. Find and/or annotate the dataset you'll train your model on.
4. Tokenize the dataset.

### 1.1 Requirements

The SongGen code is written in [PyTorch](https://pytorch.org) and [Accelerate](https://huggingface.co/docs/accelerate/index). It uses some additional requirements, like [wandb](https://wandb.ai/), especially for logging and evaluation.

Install the package for training.

```bash
pip install -e .[train]
```

Optionally, you can create a wandb account and login to it by following [this guide](https://docs.wandb.ai/quickstart). [`wandb`](https://docs.wandb.ai/) allows for better tracking of the experiments metrics and losses.

You also have the option to configure Accelerate by running the following command. Note that you should set the number of GPUs you wish to use for training, and also the data type (dtype) to your preferred dtype for training/inference (e.g. `bfloat16` on A100 GPUs, `float16` on V100 GPUs, etc.):

```bash
accelerate config
```

Lastly, you can link you Hugging Face account so that you can push model repositories on the Hub. This will allow you to save your trained models on the Hub so that you can share them with the community. Run the command:

```bash
git config --global credential.helper store
huggingface-cli login
```
And then enter an authentication token from https://huggingface.co/settings/tokens. Create a new token if you do not have one already. You should make sure that this token has "write" privileges.

### 1.2 Initialize a model from scratch or use a pre-trained one.

Depending on your compute resources and your dataset, you need to choose between fine-tuning a pre-trained model and training a new model from scratch.

In that sense, we released checkpoints: [`LiuZH-19/SongGen_mixed_pro`](https://huggingface.co/LiuZH-19/SongGen_mixed_pro) and [`LiuZH-19/SongGen_interleaving_A_V`](https://huggingface.co/LiuZH-19/SongGen_mixed_pro), that you can fine-tune for your own use-case.

You can also train you own model from scratch. You can find [here](/helpers/model_init_scripts/) examples on how to initialize a model from scratch. For example, you can initialize a mixed_pro mode model with:

```sh
python helpers/model_init_scripts/init_model_mixed.py ./outputs/untrained-mixed_pro  --text_model "google/flan-t5-large"  --track_pattern mixed_pro
```

### 1.3 Create or find datasets

To train your own SongGen model, you need a dataset with the following main features:

- Song audio
- Lyrics
- Text description
- Duration
- Separated audio tracks ( for dual-track training )
    - Vocal
    - Accompaniment 
- Optional: Other tracks for extension

**Note**: Our preprocessing pipeline is under development and will be released soon.
We have already released the MusicCaps test set (see [`LiuZH-19/MusicCaps_Test_Song`](https://huggingface.co/datasets/LiuZH-19/MusicCaps_Test_Song)), which you can refer to for the recommended data format. Here we use this dataset as a simple example for preparing training data for our model.

**Example:** Preparing MusicCaps Test Set

Download  the dataset.

```
git lfs install
git clone https://huggingface.co/datasets/LiuZH-19/MusicCaps_Test_Song

```
Because the original metadata.json stores audio paths as relative paths, you may need to convert them to absolute paths for compatibility with downstream processing.
Here is an example Python script:

```

import os
import json

input_jsonl = "datasets/MusicCaps_Test_Song/metadata.jsonl"
output_jsonl = "datasets/MusicCaps_Test_Song/metadata_abspath.jsonl"
root_dir = "/fs-computility/mllm/liuzihan/SongGen/datasets/MusicCaps_Test_Song"

def add_abspaths_to_item(item):
    for k, v in list(item.items()):
        if k.endswith("file_name"):
            if not os.path.isabs(v):
                abspath = os.path.abspath(os.path.join(root_dir, v))
            else:
                abspath = v
            atype = k.split('_')[0]
            item[ atype+ "_abspath"] = abspath
    return item

with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        item = add_abspaths_to_item(item)
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Absolute paths have been added for all fields ending with 'file_name'. Output saved to", output_jsonl)

```

After processing, each item in the dataset should have the following structure:

```json

{
  "fname": "",  
  "caption": "...",
  "lyrics": "...",
  "audio_abspath": "/XXX/XXX.wav",
  "vocal_abspath": "/XXX/XXX.wav",
  "acc_abspath": "/XXX/XXX.wav",
  "vocal_energy": 1066.88,
  "acc_energy": 895.92,
  "language": "english",
  "duration": 9.9935
}

```
### 1.4. Tokenize the audio and text data

Since audio tokenization is costly and time-consuming, we preprocess and save the tokenized data in advance to avoid repeating this step during each training run.

The script [`preprocess_data.py`](/training/preprocess_data.py) performs the following steps:
1. Load dataset(s) and merge them to the annotation dataset(s) if necessary.
2. Tokenize the lyrics and caption text.
3. Pre-computes audio tokens for both target and reference audio.
    **Note**:  In this script, we use the mixed audio as the target audio, and the vocal and accompaniment tracks as reference audio. This is primarily to process and store all audio tracks simultaneously. Please note that the `ref_audio_column_name` parameter is used only for data preparation, and its meaning differs from its usage during model training.
4. Save the processed data to the specified "save_to_disk" directory.


we provide an example JSON config file [`data_musiccaps.json`](/helpers/training_configs/data_musiccaps.json) for MusicCaps. To process the MusicCaps dataset with this configuration, execute the following command:

```sh
accelerate launch ./training/preprocess_data.py  ./helpers/training_configs/data_musiccaps.json
```

## 2. Training

To train SongGen Mix mode, use the following commands:

```sh
export training_config=./helpers/training_configs/mixed_pro_step1.json
# export training_config=./helpers/training_configs/mixed_pro_step2_a.json
# export training_config=./helpers/training_configs/mixed_pro_step2_b.json
# export training_config=./helpers/training_configs/mixed_pro_step3.json
accelerate launch ./training/run_songgen_training_mixed.py $training_config
```

To train SongGen Dual-track mode, first copy the  `mixed_pro/step1` model weights to `inter_A_V/step1` to initialize the dual-track model.
Then, use the following commands to train in Dual-Track mode:

```sh

export training_config=./helpers/training_configs/inter_A_V_step1_5.json
# export training_config=./helpers/training_configs/inter_A_V_step2_a.json
# export training_config=./helpers/training_configs/inter_A_V_step2_b.json
# export training_config=./helpers/training_configs/inter_A_V_step3.json
accelerate launch ./training/run_songgen_training_dual_track.py $training_config

```
All the configuration files referenced above are already included in the `helpers/training_configs` directory.

> [!TIP]
> Fine-tuning is as easy as modifying `model_name_or_path` to a pre-trained model.
> For example: `--model_name_or_path LiuZH-19/SongGen_mixed_pro`.

Our codebase is based on [ParlerTTS](https://github.com/huggingface/parler-tts), with data preprocessing and training code organized into separate files for clarity. For further implementation details, you may also refer to the ParlerTTS documentation.
