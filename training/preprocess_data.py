#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Preprocess SongGen Data"""
import pdb
import logging
import os
import re
import sys

import time
from multiprocess import set_start_method
from datetime import timedelta

from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets, enable_caching, disable_caching 
from datasets.features import Audio
import tempfile
from huggingface_hub import HfApi

import transformers
from transformers import  AutoTokenizer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.optimization import get_scheduler
from transformers.utils import send_example_telemetry
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor


from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs, TorchDynamoPlugin, DistributedDataParallelKwargs
from accelerate.utils.memory import release_memory

# from training.custom_audio_feature import MyAudio

from songgen import (
    build_delay_pattern_mask,
    VoiceBpeTokenizer,
    SongGenMixedForConditionalGeneration,
    SongGenConfig,
    XCodecModel
)


from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred,
    log_metric,
    log_gt,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from training.arguments import ModelArguments, DataTrainingArguments, SongGenTrainingArguments
from training.data import load_multiple_datasets, DataCollatorCodecWithPadding
from training.eval import clap_similarity, wer, si_sdr
import numpy as np
import random
from my_proxy import *


logger = logging.getLogger(__name__)
storage_options = {"endpoint_url":'', "key":'', "secret": '', "use_ssl": False}




def judge_was_precomputed(save_path):
    if isinstance(save_path, list):
        save_path = save_path[0] #Only take the first one for inspection
    if save_path is None:
        logger.warning('save path is None. Please define the save path.')
        return False
    else:
        if save_path.startswith('s3://'):
            pass
        else:
            os.makedirs(save_path, exist_ok=True)
            # assume that the dataset has been saved to `save_to_disk` if the latter is not empty
            dataset_was_precomputed = len(os.listdir(save_path)) > 0
        return  dataset_was_precomputed
    

def save_dataset_to_disk(dataset, save_path, num_proc=16):
    if save_path.startswith('s3://'):
        disable_proxy()
        dataset.save_to_disk(save_path, storage_options=storage_options, num_proc=num_proc)
        enable_proxy()
    else:
        dataset.save_to_disk(save_path, num_proc=num_proc)



def load_dataset_from_disk(save_path):    
    if save_path.startswith('s3://'):
        disable_proxy()
        dataset = datasets.load_from_disk(save_path, storage_options=storage_options)
        enable_proxy()
    else:
        dataset = datasets.load_from_disk(save_path)
    return dataset



def main():
    # tempfile.tempdir ='../datasets/tmp' 
    enable_caching()
    enable_proxy()


    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SongGenTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_songgen", model_args, data_args)

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        torch_dtype = torch.float32

    if data_args.pad_to_max_length and (
        data_args.max_duration_in_seconds is None
        or data_args.max_prompt_token_length is None
        or data_args.max_description_token_length is None
    ):
        raise ValueError(
            "`pad_to_max_length` is `True` but one of the following parameters has not been set: `max_duration_in_seconds`, `max_prompt_token_length`, `max_description_token_length`"
        )

    padding = "max_length" if data_args.pad_to_max_length else "longest"

    ####### A. Preparation
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120))]

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=training_args.report_to,
        project_dir=training_args.output_dir,
        kwargs_handlers=kwargs_handlers,
    )

    accelerator.init_trackers(
        project_name=data_args.wandb_project,
        config={
            "learning_rate": training_args.learning_rate,
            "model_name_or_path": model_args.model_name_or_path,
            "num_train_epochs": training_args.num_train_epochs,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "global_batch_size": training_args.per_device_train_batch_size * accelerator.num_processes,
            "mixed_precision": mixed_precision,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_steps": training_args.warmup_steps,
            "freeze_text_encoder": model_args.freeze_text_encoder,
            "freeze_cross": model_args.freeze_cross,
            "max_duration_in_seconds": data_args.max_duration_in_seconds,
            "weight_decay": training_args.weight_decay,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "temperature": model_args.temperature,
        },
        init_kwargs={"wandb": {"name": data_args.wandb_run_name}} if data_args.wandb_run_name else {},
    )

    # Detecting last checkpoint and eventually continue from last checkpoint
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.WARN)

    # Log a small summary on each proces
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    num_workers = data_args.preprocessing_num_workers

    # 1. First, lett's instantiate the feature extractor, tokenizers and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # load feature extractor
    # feature_extractor = AutoFeatureExtractor.from_pretrained(
    #     model_args.feature_extractor_name or model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     token=data_args.token,
    #     trust_remote_code=data_args.trust_remote_code,
    # )
    # sampling_rate = feature_extractor.sampling_rate

    sampling_rate =  16000   # X-codec sampling-rate
    feature_extractor =  EncodecFeatureExtractor(sampling_rate=sampling_rate)

    voicebpe_tokenizer = VoiceBpeTokenizer()

    # load prompt tokenizer
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        model_args.prompt_tokenizer_name or model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side=model_args.prompt_padding_side
    )

    # load description tokenizer
    description_tokenizer = AutoTokenizer.from_pretrained(
        model_args.description_tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )

    if model_args.use_fast_tokenizer:
        logger.warning(
            "Disabling fast tokenizer warning: https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3231-L3235"
        )
        prompt_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        description_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # 2. Now, let's load the dataset
    dataset_was_precomputed=judge_was_precomputed(data_args.save_to_disk)

    assert not dataset_was_precomputed, f"Dataset has already been preprocessed and saved to {data_args.save_to_disk}; preprocessing steps need not be executed again."
    assert data_args.preprocessing_only and data_args.save_to_disk is not None,  "When running in preprocessing-only mode, `data_args.save_to_disk` must be specified."

   
            
    raw_datasets = DatasetDict()

    # prepare columns_to_keep
    columns_to_keep = {
        "target_audio_column_name": data_args.target_audio_column_name,
        "prompt_column_name": data_args.prompt_column_name,
        "id": data_args.id_column_name,
        # "clap_score": "clap_score",
        # "in_vad":"in_vad",
        # "edit_dist_v12": "edit_dist_v12"

    }
    if data_args.description_column_name is not None:
        columns_to_keep["description_column_name"] = data_args.description_column_name
    
    if data_args.ref_voice_column_name is not None:
        columns_to_keep["ref_voice_column_name"] = data_args.ref_voice_column_name

    if data_args.ref_audio_column_name is not None:
        if isinstance(data_args.ref_audio_column_name, list):
            for name in data_args.ref_audio_column_name:
                atype = name.split('_')[0]
                columns_to_keep[f"ref_audio_{atype}_column_name"] = name
                columns_to_keep[f"{atype}_energy"] = f"{atype}_energy" # Optional: used for filtering audio samples based on energy in DataCollatorSongGenWithPadding
        # if data_args.ref_audio_column_name != 'ALL':
        #     columns_to_keep["ref_audio_column_name"] = data_args.ref_audio_column_name
        # else:
        #     for atype in ['acc', 'vocal', 'drum']:
        #         columns_to_keep[f"ref_audio_{atype}_column_name"] = f"{atype}_abspath"
        #         columns_to_keep[f"{atype}_clip_energy"] = f"{atype}_clip_energy"
                


    

    # load dataset
    if training_args.do_train:            
        raw_datasets["train"] = load_multiple_datasets(
            accelerator,
            data_args.loading_method,
            data_args.train_dataset_name,
            data_args.train_dataset_config_name,
            metadata_dataset_names=data_args.train_metadata_dataset_name,
            splits=data_args.train_split_name,
            dataset_samples=data_args.train_dataset_samples,
            seed=training_args.seed,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            id_column_name=data_args.id_column_name,
            columns_to_keep=columns_to_keep.values(),
            prompt_column_name=data_args.prompt_column_name,
            audio_column_name=data_args.target_audio_column_name,
            sampling_rate=sampling_rate,
            logger=logger,
            # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
        )

        for key in columns_to_keep:
            if columns_to_keep[key] not in raw_datasets["train"].column_names:
                raise ValueError(
                    f"--{key} '{columns_to_keep[key]}' not found in dataset '{data_args.train_dataset_name}'."
                    f" Make sure to set `--{key}` to the correct audio column - one of"
                    f" {', '.join(raw_datasets['train'].column_names)}."
                )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"] = load_multiple_datasets(
            accelerator,
            data_args.loading_method,
            data_args.eval_dataset_name if data_args.eval_dataset_name else data_args.train_dataset_name,
            data_args.eval_dataset_config_name
            if data_args.eval_dataset_config_name
            else data_args.train_dataset_config_name,
            metadata_dataset_names=data_args.eval_metadata_dataset_name,
            splits=data_args.eval_split_name,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            id_column_name=data_args.id_column_name,
            columns_to_keep=columns_to_keep.values(),
            prompt_column_name=data_args.prompt_column_name,
            audio_column_name=data_args.target_audio_column_name,
            sampling_rate=sampling_rate,
            logger=logger,
            # streaming=data_args.streaming, TODO(SG): optionally enable streaming mode
        )
        
        if data_args.max_eval_samples is not None:
            with accelerator.local_main_process_first():
                raw_datasets["eval"] = (
                    raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
                )
    logger.info(f"Finished loading the dataset: {raw_datasets}")
            

    # 3. Next, let's load the config.
    config = SongGenConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # update pad token id and decoder_start_token_id
    
    config.update(
        {
            "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else config.decoder_start_token_id,
        }
    )

    # create model
    model = SongGenMixedForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )
    model.audio_encoder = XCodecModel()
    


   
    

    # 4. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # derive max & min input length for sample rate & max duration
    sampling_rate = feature_extractor.sampling_rate
    max_target_length = int(data_args.max_duration_in_seconds * sampling_rate)
    min_target_length = int(data_args.min_duration_in_seconds * sampling_rate)
    target_audio_column_name = data_args.target_audio_column_name
    description_column_name = data_args.description_column_name
    ref_voice_column_name = data_args.ref_voice_column_name
    ref_audio_column_name = data_args.ref_audio_column_name
    prompt_column_name = data_args.prompt_column_name
    feature_extractor_input_name = feature_extractor.model_input_names[0]
    audio_encoder_pad_token_id = config.decoder.pad_token_id
    audio_encoder_eos_token_id = config.decoder.eos_token_id
    audio_encoder_bos_token_id = model.generation_config.decoder_start_token_id
    max_length = model.generation_config.max_length
    num_codebooks = model.decoder.config.num_codebooks
    bandwidth = model_args.bandwidth
    attn_implementation = model_args.attn_implementation

    audio_encoder_special_token_ids ={
        'mask': 1030,
        "random": {"bos": 1031 , "eos": 1032}, 
        "melody": {"bos": 1033, "eos": 1034},
        "drum": {"bos": 1035, "eos": 1036},
        "vocal": {"bos": 1037, "eos": 1038},
        "acc": {"bos":1039 , "eos":1040},
    }


   
    # Test all gather - used for warmout and avoiding timeout
    logger.debug(str(accelerator.process_index), main_process_only=False, in_order=True)
    test_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
    gathered_tensor = accelerator.gather(test_tensor)
    print("gathered_tensor", gathered_tensor)
    accelerator.wait_for_everyone()
    

    if ref_voice_column_name is not None:
        # process inputs for mert music representation model
        from transformers import Wav2Vec2FeatureExtractor
        import torchaudio.transforms as T
        mert_path='m-a-p/MERT-v1-330M'
        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path, 
            cache_dir=model_args.cache_dir,    
            trust_remote_code=True) 
        refvoice_sr = mert_processor.sampling_rate
        ref_dur_sec = data_args.ref_dur_sec if data_args.ref_dur_sec is not None else 3
    else:
        mert_processor =None
        ref_dur_sec = 0


    with accelerator.local_main_process_first():
        raw_datasets = raw_datasets.cast_column(target_audio_column_name, Audio(sampling_rate=sampling_rate))

    # Filter on text length
    if description_column_name is not None and data_args.max_text_length is not None:
        with accelerator.local_main_process_first():
            # filter description that is shorter than max_text_length
            raw_datasets = raw_datasets.filter(
                lambda x: len(x) < data_args.max_text_length,
                num_proc=num_workers,
                input_columns=[description_column_name],
            )

    # Preprocessing the dataset.
    # We need to tokenize the description texts and lyrics.
    def pass_through_processors(description, prompt):
        batch = {}
        batch["input_ids"] = description_tokenizer(description.strip())["input_ids"]
        if prompt.strip()=='':
            batch["prompt_input_ids"] = [261, 0]
        else:
            batch["prompt_input_ids"] = [261]+ voicebpe_tokenizer.encode(prompt.strip().replace('\n', '.'), lang='en') +[0]
        return batch
    
    all_column_names = next(iter(raw_datasets.values())).column_names
    vect_to_keep_cols = [data_args.id_column_name] + [col for col in all_column_names if col.endswith('_energy')] #[in_vad]
    vect_to_remove = [col for col in all_column_names if col not in vect_to_keep_cols]

    with accelerator.local_main_process_first():
        # this is a trick to avoid to rewrite the entire audio column which takes ages
        vectorized_datasets = raw_datasets.map(
            pass_through_processors,
            remove_columns=vect_to_remove,
            input_columns=[description_column_name, prompt_column_name],
            num_proc=num_workers,
            desc="preprocess datasets",
        )
        logger.info('After processors:', vectorized_datasets)
    

    if ref_voice_column_name is not None:    
        with accelerator.local_main_process_first():
            logger.info('To preprocess Mert inputs!!')
            mert_datasets = raw_datasets.select_columns([ref_voice_column_name]).rename_columns({ref_voice_column_name: 'mert_input'})

    

    if ref_audio_column_name is not None and ref_audio_column_name != target_audio_column_name:   
        with accelerator.local_main_process_first():
            if isinstance(ref_audio_column_name, str):
                raw_datasets = raw_datasets.cast_column(ref_audio_column_name, Audio(sampling_rate=sampling_rate))
                logger.info(f'Processed single ref_audio_column: {ref_audio_column_name}')
            elif isinstance(ref_audio_column_name, list):
                for ref_col in ref_audio_column_name:
                    raw_datasets = raw_datasets.cast_column(ref_col, Audio(sampling_rate=sampling_rate))
                    logger.info(f'Processed ref_audio_column list: {ref_col}')
            else:
                raise ValueError(f"Invalid type for ref_audio_column: {type(ref_audio_column_name)}. Expected str or list.")
    # We use Accelerate to perform distributed inference
    # T5 doesn't support fp16
    autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))

    # Now we encode the audio labels with encodec.
    ####### B. Encode audio

    logger.info("*** Encode target audio with X-codec ***")
    # no need to prepare audio_decoder because used for inference without mixed precision
    # see: https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.prepare
    if training_args.torch_compile:
        audio_decoder = accelerator.prepare_model(model.audio_encoder, evaluation_mode=True)
    else:
        audio_decoder = model.audio_encoder
    encoder_data_collator = DataCollatorCodecWithPadding(
        feature_extractor,
        audio_column_name=target_audio_column_name,
        ref_audio_column_name=ref_audio_column_name,
        feature_extractor_input_name=feature_extractor_input_name,
        max_length=max_target_length,
        padding=padding,
    )
        

    def apply_audio_decoder(batch):
        assert len(batch['input_values']) == 1, f"apply_audio_decoder batch_size: {len(batch['input_values'])}"
        len_audio = batch.pop("len_audio")
        audio_decoder.to(batch["input_values"].device).eval()
        
        with torch.no_grad():
            wav = batch['input_values'][0]
            wav = wav.unsqueeze(1)
            assert len(wav.shape)==3 and wav.shape[0] == 1 and wav.shape[1] == 1 ,  f'apply_audio_decoder wav : {wav.shape}'
            labels = audio_decoder.encode(wav)["audio_codes"].squeeze(0).squeeze(0) #(1, bsz, codebooks, seq_len) ->(codebooks, seq_len)
        output = {}
        output["len_audio"] = len_audio
        # (codebooks,  seq_len) -> (1,  seq_len, codebooks)
        output["labels"] = labels.transpose(1, 0).unsqueeze(0) 

        # if `pad_to_max_length`, the maximum corresponding audio length of the current batch is max_duration*sampling_rate
        max_length = len_audio.max() if padding != "max_length" else max_target_length
        output["ratio"] = torch.ones_like(len_audio) * labels.shape[-1] / max_length
        return output

    # (1, codebooks, seq_len) where seq_len=1
    bos_labels = torch.ones((1, num_codebooks, 1)) * audio_encoder_bos_token_id

    def postprocess_dataset(labels):
        # (1, codebooks, seq_len)
        labels = torch.tensor(labels).unsqueeze(0)
        # add bos
        labels = torch.cat([bos_labels, labels], dim=-1)

        labels, delay_pattern_mask = build_delay_pattern_mask(
            labels,
            bos_token_id=audio_encoder_bos_token_id,
            pad_token_id=audio_encoder_eos_token_id,
            max_length=labels.shape[-1] + num_codebooks,
            num_codebooks=num_codebooks,
        )

        # the first ids of the delay pattern mask are precisely labels, we use the rest of the labels mask
        # to take care of EOS
        # we want labels to look like this:
        #  - [B, a, b, E, E, E, E]
        #  - [B, B, c, d, E, E, E]
        #  - [B, B, B, e, f, E, E]
        #  - [B, B, B, B, g, h, E]
        labels = torch.where(delay_pattern_mask == -1, audio_encoder_eos_token_id, delay_pattern_mask)

        # the first timestamp is associated to a row full of BOS, let's get rid of it
        # we also remove the last timestampts (full of PAD)
        output = {"labels": labels[:, 1:]}
        return output
    
    def postprocess_refaudios(labels, atype):
        bos = audio_encoder_special_token_ids[atype]['bos']
        eos = audio_encoder_special_token_ids[atype]['eos']
        # ref_audio_encoder_bos_token_id = 1031
        # ref_audio_encoder_eos_token_id = 1032
        ref_bos_labels = torch.ones((1, num_codebooks, 1)) * bos
        # (1, codebooks, seq_len)
        labels = torch.tensor(labels).unsqueeze(0)
        # add bos
        labels = torch.cat([ref_bos_labels, labels], dim=-1)

        labels, delay_pattern_mask = build_delay_pattern_mask(
            labels,
            bos_token_id=bos,
            pad_token_id=eos,
            max_length=labels.shape[-1] + num_codebooks,
            num_codebooks=num_codebooks,
        )

        # the first ids of the delay pattern mask are precisely labels, we use the rest of the labels mask
        # to take care of EOS
        # we want labels to look like this:
        #  - [B, a, b, E, E, E, E]
        #  - [B, B, c, d, E, E, E]
        #  - [B, B, B, e, f, E, E]
        #  - [B, B, B, B, g, h, E]
        labels = torch.where(delay_pattern_mask == -1, eos, delay_pattern_mask) 

        #TO DEL # the first timestamp is associated to a row full of BOS, let's get rid of it
        # we also remove the last timestampts (full of PAD)
        output = {f"ref_audio_ids_{atype}": labels} #[:, 1:] 第一排全为ref_bos
        return output

    def init_all_ref(atype_list=['acc', 'vocal', 'drum']):
        all_refaudios ={}
        all_refaudio_lens = {}
        for at in atype_list:
            all_refaudios[at] = []
            all_refaudio_lens[at] = []
        return all_refaudios, all_refaudio_lens

    atype_list = [col.split('_')[0] for col in ref_audio_column_name]
    for split in vectorized_datasets:
        data_loader = DataLoader(
            raw_datasets[split],
            batch_size=training_args.audio_encoder_per_device_batch_size,
            collate_fn=encoder_data_collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
        )
        data_loader = accelerator.prepare(data_loader)
        total_inference_steps = len(data_loader)

        start_step = get_last_codec_checkpoint_step(os.path.join(data_args.temporary_save_to_disk, split))
        accelerator.wait_for_everyone()
        if start_step > 0:
            logger.info(f"Resuming {split} from step {start_step}")
            # efficiently skip the first n batches
            start_step += 1
            data_loader = skip_first_batches(data_loader, start_step)

        all_generated_labels = []
        all_lens = []

        all_refaudios, all_refaudio_lens =init_all_ref(atype_list)

        if start_step < total_inference_steps:
            for i, (batch, refaudio_batch) in enumerate(tqdm(data_loader, disable=not accelerator.is_local_main_process)):
                cur_step = start_step + i
                generate_labels = apply_audio_decoder(batch)
                generate_labels = accelerator.pad_across_processes(generate_labels, dim=1, pad_index=0)
                generate_labels = accelerator.gather_for_metrics(generate_labels)

                generate_refaudios_all = {}
                for atype in refaudio_batch.keys():
                    generate_refaudios = apply_audio_decoder(refaudio_batch[atype])
                    generate_refaudios = accelerator.pad_across_processes(generate_refaudios, dim=1, pad_index=0)
                    generate_refaudios = accelerator.gather_for_metrics(generate_refaudios)
                    generate_refaudios_all[atype] = generate_refaudios

                if accelerator.is_main_process:
                    lab = generate_labels["labels"].cpu().transpose(1, 2).to(torch.int16)
                    rat = generate_labels["ratio"].cpu().squeeze(1)
                    lens = generate_labels["len_audio"].cpu().squeeze(1)
                    lab = [l[:, : int(ratio * length)] for (l, ratio, length) in zip(lab, rat, lens)]

                    all_generated_labels.extend(lab)
                    all_lens.extend(lens)

                    for atype in generate_refaudios_all.keys():
                        ref_lab = generate_refaudios_all[atype]["labels"].cpu().transpose(1, 2).to(torch.int16)
                        ref_rat = generate_refaudios_all[atype]["ratio"].cpu().squeeze(1)
                        ref_lens = generate_refaudios_all[atype]["len_audio"].cpu().squeeze(1)
                        ref_lab = [l[:, : int(ratio * length)] for (l, ratio, length) in zip(ref_lab, ref_rat, ref_lens)]

                        all_refaudios[atype].extend(ref_lab)
                        all_refaudio_lens[atype].extend(ref_lens)


                    if ((cur_step + 1) % data_args.save_codec_steps == 0) or (
                        cur_step == total_inference_steps - 1
                    ):
                        tmp_labels = Dataset.from_dict({"labels": all_generated_labels, "target_length": all_lens})
                        tmp_labels = tmp_labels.map(
                            postprocess_dataset,
                            num_proc=data_args.preprocessing_num_workers,  # this one is resource consuming if many processor.
                            input_columns=["labels"],
                            desc="Postprocessing labeling",
                        )
                        for atype in all_refaudios.keys():
                            tmp_refaudios = Dataset.from_dict({f"ref_audio_ids_{atype}": all_refaudios[atype], f"refaudio_length_{atype}": all_refaudio_lens[atype]})
                            tmp_refaudios = tmp_refaudios.map(
                                lambda x: postprocess_refaudios(x, atype),
                                num_proc=data_args.preprocessing_num_workers,  # this one is resource consuming if many processor.
                                input_columns=[f"ref_audio_ids_{atype}"],
                                desc=f"Postprocessing refaudios: {atype}",
                            )
                            
                            tmp_labels = concatenate_datasets([tmp_labels, tmp_refaudios], axis=1)
                                
                        save_codec_checkpoint(
                            os.path.join(data_args.temporary_save_to_disk, split), tmp_labels, cur_step
                        )
                        all_generated_labels = []
                        all_lens = []
                        all_refaudios, all_refaudio_lens =init_all_ref(atype_list)

            accelerator.wait_for_everyone()

        if accelerator.is_main_process and len(all_generated_labels) > 0:
            tmp_labels = Dataset.from_dict({"labels": all_generated_labels, "target_length": all_lens})
            tmp_labels = tmp_labels.map(
                postprocess_dataset,
                num_proc=data_args.preprocessing_num_workers,  # this one is resource consuming if many processor.
                input_columns=["labels"],
                desc="Postprocessing labeling",
            )

            for atype in all_refaudios.keys():
                tmp_refaudios = Dataset.from_dict({f"ref_audio_ids_{atype}": all_refaudios[atype], f"refaudio_length_{atype}": all_refaudio_lens[atype]})
                tmp_refaudios = tmp_refaudios.map(
                    lambda x: postprocess_refaudios(x, atype),
                    num_proc=data_args.preprocessing_num_workers,  # this one is resource consuming if many processor.
                    input_columns=[f"ref_audio_ids_{atype}"],
                    desc=f"Postprocessing refaudios: {atype}",
                )
                tmp_labels = concatenate_datasets([tmp_labels, tmp_refaudios], axis=1)
                    
            save_codec_checkpoint(
                os.path.join(data_args.temporary_save_to_disk, split), tmp_labels, cur_step
            )
            all_generated_labels = []
            all_lens = []
            all_refaudios, all_refaudio_lens =init_all_ref(atype_list)

        accelerator.wait_for_everyone()

        del all_generated_labels, all_refaudios
        accelerator.wait_for_everyone()

        with accelerator.local_main_process_first():
            tmp_labels_refaudios = load_all_codec_checkpoints(os.path.join(data_args.temporary_save_to_disk, split)).select(
                range(len(vectorized_datasets[split]))
            )
            
            if ref_voice_column_name is not None:
                logger.info(f"Concatenating {split}: {tmp_labels_refaudios} , {vectorized_datasets[split]} , {mert_datasets[split]} ")
                vectorized_datasets[split] = concatenate_datasets([vectorized_datasets[split], mert_datasets[split], tmp_labels_refaudios], axis=1)
            else:
                logger.info(f"Concatenating {split}: {tmp_labels_refaudios} with {vectorized_datasets[split]}")
                vectorized_datasets[split] = concatenate_datasets([vectorized_datasets[split], tmp_labels_refaudios], axis=1)


    accelerator.free_memory()
    del  all_lens,  all_refaudio_lens 

    with accelerator.local_main_process_first():
        # NOTE: filtering is done at the end because in the `datasets` library, caching audio files is done after most operations
        # caching audio files is time and disk-space consuming, so we want to avoid it at all costs, especially for large (>1Kh) audio datasets.
        # That's also why we avoid to concat the processed datasets (vectorized_datasets) with the audio column present in raw_datasets.

        def is_audio_in_length_range(length):
            return length > min_target_length and length < max_target_length

        # filter data that is shorter than min_target_length
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["target_length"],
        )

    if description_column_name is not None and data_args.max_description_token_length is not None:
        with accelerator.local_main_process_first():
            # filter description that is shorter than max_text_length
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: len(x) < data_args.max_description_token_length,
                num_proc=num_workers,
                input_columns=["input_ids"],
            )

    if data_args.max_prompt_token_length is not None:
        with accelerator.local_main_process_first():
            # filter description that is shorter than max_text_length
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: len(x) < data_args.max_prompt_token_length,
                num_proc=num_workers,
                input_columns=["prompt_input_ids"],
                )

    if data_args.save_to_disk is not None:
        if accelerator.is_main_process:
            logger.info(f"!!!To save Dataset at {data_args.save_to_disk}")
            save_dataset_to_disk(vectorized_datasets, data_args.save_to_disk, num_proc=min(data_args.preprocessing_num_workers, len(vectorized_datasets["eval"]) - 1))
            
        accelerator.wait_for_everyone()
        logger.info(f"Data preprocessing finished. Files save at {data_args.save_to_disk}")
        return
    
    logger.info(f"Finally Clean Dataset: {vectorized_datasets}")

    



if __name__ == "__main__":
    main()
