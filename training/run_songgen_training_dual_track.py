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

""" Train SongGen using 🤗 Accelerate"""
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
from accelerate.utils import set_seed, AutocastKwargs, InitProcessGroupKwargs
from accelerate.utils.memory import release_memory

from songgen import (
    build_delay_pattern_mask,
    VoiceBpeTokenizer,
    SongGenDualTrackForConditionalGeneration,
    SongGenConfig,
    XCodecModel,
    combine_track_input_ids,
    build_delay_pattern_mask,
    split_combined_track_input_ids
)
from training.utils import (
    get_last_checkpoint,
    rotate_checkpoints,
    log_pred_AV,
    log_metric,
    log_gt,
    load_all_codec_checkpoints,
    save_codec_checkpoint,
    get_last_codec_checkpoint_step,
)
from training.arguments import ModelArguments, DataTrainingArguments, SongGenTrainingArguments
from training.data import DataCollatorSongGenWithPadding
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


def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6  


def main():
    enable_proxy()
    enable_caching()
    # tempfile.tempdir ='../datasets/tmp'
                 
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
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(minutes=120))] # DistributedDataParallelKwargs(find_unused_parameters=True)

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
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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
    feature_extractor =  EncodecFeatureExtractor(sampling_rate = sampling_rate)
    
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
    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    dataset_was_precomputed=judge_was_precomputed(data_args.save_to_disk)
    assert dataset_was_precomputed, f"Dataset has not been preprocessed, please run `preprocess_data.py` first to preprocess data and save to {data_args.save_to_disk}"

    with accelerator.local_main_process_first():
        if not isinstance(data_args.save_to_disk, list):
            logger.info(f'load dataset: {data_args.save_to_disk}')
            vectorized_datasets = load_dataset_from_disk(data_args.save_to_disk)
        else:
            vectorized_datasets = DatasetDict()
            train_data_list = []
            eval_data_list = []
            for data_path in data_args.save_to_disk:
                logger.info(f'load dataset: {data_path}')
                tmp_datasets = load_dataset_from_disk(data_path)
                train_data_list.append(tmp_datasets['train'])
                eval_data_list.append(tmp_datasets['eval'])
            vectorized_datasets['train'] = concatenate_datasets(train_data_list, axis=0)
            vectorized_datasets['eval'] = concatenate_datasets(eval_data_list, axis=0)


        if data_args.max_train_samples is not None:
            vectorized_datasets["train"] = (
                    vectorized_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
                )
        if data_args.max_eval_samples is not None:
            vectorized_datasets["eval"] = (
                        vectorized_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
                    )
           
           


    # 3. Next, let's load the config.
    config = SongGenConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # update pad token id and decoder_start_token_id
    config.decoder.update(
        {
            "track_pattern": model_args.track_pattern,  
            "add_vocal_loss": training_args.add_vocal_loss,
            "cross_attention_implementation_strategy": model_args.cross_attention_implementation_strategy
            if model_args.cross_attention_implementation_strategy is not None
            else None
        }
    )
    config.update(
        {
            "pad_token_id": model_args.pad_token_id if model_args.pad_token_id is not None else config.pad_token_id,
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else config.decoder_start_token_id,
            "prompt_cross_attention": model_args.prompt_cross_attention,
            "add_prenet": model_args.add_prenet if model_args.add_prenet is not None else True
        }
    )

    # create model
    model = SongGenDualTrackForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        # ignore_mismatched_sizes=True, 
    )
    model.audio_encoder = XCodecModel()

    # In case one passes a config to `from_pretrained` + "attn_implementation"
    # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs


    # enable gradient checkpointing if necessary
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        


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

    #NOTE
    audio_encoder_special_token_ids ={
        'mask': 1030,
        "random": {"bos": 1031 , "eos": 1032}, 
        "melody": {"bos": 1033, "eos": 1034},
        "drum": {"bos": 1035, "eos": 1036},
        "vocal": {"bos": 1037, "eos": 1038},
        "acc": {"bos":1039 , "eos":1040},
    }


    # Freeze 
    model.freeze_encoders(model_args.freeze_text_encoder)
    model.freeze_cross(model_args.freeze_cross)
    model.freeze_embed_prompts(model_args.freeze_embed_prompts)

    logger.info(f'embed_prompts requires_grad:{any(param.requires_grad for param in model.embed_prompts.parameters())}')  
    logger.info(f'audio_embed_tokens requires_grad:{any(param.requires_grad for param in model.decoder.model.decoder.embed_tokens.parameters())}') 
    logger.info(f'trainable_parmameters numbers: {count_trainable_parameters(model)} M') 
    logger.info(f'trainable_parmameters in lyrics_encoder : {count_trainable_parameters(model.lyrics_encoder)} M') 
    logger.info(f'trainable_parmameters in audio_encoder : {count_trainable_parameters(model.audio_encoder)} M') 
    logger.info(f'trainable_parmameters in transformer decoder : {count_trainable_parameters(model.decoder)} M') 


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
        refvoice_sr = 24000
        

    
    audio_max_length = None
    if padding == "max_length":
        audio_max_length = max(vectorized_datasets["train"]["target_length"])
        with accelerator.local_main_process_first():
            max_sample = vectorized_datasets["train"].filter(
                lambda x: x == audio_max_length,
                num_proc=num_workers,
                input_columns=["target_length"],
            )
        audio_max_length = max([len(l[0]) for l in max_sample["labels"]])

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

    if training_args.group_by_length:
        # apply a simple heuristic to take into account audio and text lengths
        def add_target_lengths(target_length, prompt, description):
            return {"target_length": target_length + len(prompt) + len(description)}

        with accelerator.local_main_process_first():
            vectorized_datasets = vectorized_datasets.map(
                add_target_lengths,
                num_proc=num_workers*2,
                input_columns=["target_length", "prompt_input_ids", "input_ids"],
            )


    
    if ref_voice_column_name is None:
        vectorized_datasets = vectorized_datasets.remove_columns(["mert_input"])
    
    if description_column_name is None:
        vectorized_datasets = vectorized_datasets.remove_columns(["input_ids"])
    
    if ref_voice_column_name is not None and 'mert_input' in vectorized_datasets['train'].features.keys():
        vectorized_datasets = vectorized_datasets.cast_column('mert_input', Audio(sampling_rate=mert_processor.sampling_rate))
    
    logger.info(f"Finally Clean Dataset: {vectorized_datasets}")


    # 6. Next, we can prepare the training.

    # Let's use word CLAP similary and WER metrics as our evaluation metrics,
    def compute_metrics(
        acc_audios,
        vocal_audios,
        descriptions,
        prompts,
        device="cpu",
        compute_clap_similarity_metric=False,
        compute_noise_level_metric=False,
        noise_level_to_compute_clean_wer=None,
    ):
        results = {}
        input_ids = descriptions
        texts = description_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        prompts = voicebpe_tokenizer.batch_decode(prompts, skip_special_tokens=True)
        acc_audios = [a.float().cpu().numpy() for a in acc_audios]
        vocal_audios = [a.float().cpu().numpy() for a in vocal_audios]
        audios = []
        for i in range(len(acc_audios)):
            if acc_audios[i].shape[0] > vocal_audios[i].shape[0] :
                tmp = acc_audios[i].copy() #NOTE copy  不copy的话原本的acc也会被改变 之前错了！
                tmp[:vocal_audios[i].shape[0]] += vocal_audios[i]
            else:
                tmp = vocal_audios[i].copy()
                tmp[:acc_audios[i].shape[0]] += acc_audios[i] 
            audios.append(tmp)        

        if compute_clap_similarity_metric:
            clap_score = clap_similarity(
                model_args.clap_model_name_or_path, texts, audios, device, input_sampling_rate=sampling_rate
            )
            results[f"clap_{model_args.clap_model_name_or_path}"] = clap_score

        si_sdr_measures = None
        clean_word_error=None
        word_error = 100
        transcriptions = ['ignore……']*len(audios)
        
        #!!! This part of the parler-tts code is commented out, as it always causes training to terminate.
        # if compute_noise_level_metric:
        #     si_sdr_measures = si_sdr(audios, device, input_sampling_rate=sampling_rate)

        # word_error, transcriptions, clean_word_error, noisy_word_error, percent_clean_samples = wer(
        #     model_args.asr_model_name_or_path,
        #     prompts,
        #     audios,
        #     device,
        #     training_args.per_device_eval_batch_size,
        #     sampling_rate,
        #     noise_level_to_compute_clean_wer,
        #     si_sdr_measures,
        # )
        results["wer"] = word_error
        
        if clean_word_error is not None:
            results["clean_wer"] = clean_word_error
            results["noisy_word_error"] = noisy_word_error
            results["percent_clean_samples"] = percent_clean_samples

        return results, texts, prompts, acc_audios, vocal_audios, audios, transcriptions, si_sdr_measures

    # Define Training Schedule
    # Store some constants
    per_device_train_batch_size = int(training_args.per_device_train_batch_size)
    train_batch_size = per_device_train_batch_size * accelerator.num_processes
    gradient_accumulation_steps = int(training_args.gradient_accumulation_steps)
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)

    if training_args.max_steps < 0:
        num_epochs = int(training_args.num_train_epochs)
        steps_per_epoch = len(vectorized_datasets["train"]) // (train_batch_size * gradient_accumulation_steps)
        total_train_steps = steps_per_epoch * num_epochs
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        total_train_steps = int(training_args.max_steps)
        # Setting a very large number of epochs so we go as many times as necessary over the iterator.
        num_epochs = sys.maxsize
        steps_per_epoch = total_train_steps

    if training_args.eval_steps is None:
        logger.info(f"eval_steps is not set, evaluating at the end of each epoch")
        eval_steps = steps_per_epoch
    else:
        eval_steps = training_args.eval_steps

    # T5 doesn't support fp16
    autocast_kwargs = AutocastKwargs(enabled=(mixed_precision != "fp16"))

    # Define optimizer, LR scheduler, collator
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    # LR scheduler gets stepped by `num_processes` each time -> account for this in warmup / total steps
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_train_steps) * accelerator.num_processes,
        num_training_steps=total_train_steps * accelerator.num_processes,
    )

    
    data_collator = DataCollatorSongGenWithPadding(
        voicebpe_tokenizer=voicebpe_tokenizer,
        prompt_tokenizer=prompt_tokenizer,
        description_tokenizer=description_tokenizer,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
        padding=padding,
        prompt_max_length=data_args.max_prompt_token_length,
        description_max_length=data_args.max_description_token_length,
        audio_max_length=audio_max_length,
        mert_processor=mert_processor,
        ref_dur_sec=ref_dur_sec,
        ref_voice_column_name=ref_voice_column_name,
        ref_audio_column_name=ref_audio_column_name,
        audio_encoder_bos_token_id=audio_encoder_bos_token_id,
        audio_encoder_eos_token_id=audio_encoder_eos_token_id,
        audio_encoder_special_token_ids=audio_encoder_special_token_ids,
        label_atype=data_args.label_atype,
        add_vocal_labels=training_args.add_vocal_loss,
        energy_name='_clip_energy',
        track_pattern=model_args.track_pattern,
        num_codebooks=num_codebooks
    )

    # Prepare everything with accelerate
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_steps * train_batch_size * gradient_accumulation_steps}")
    logger.info("  Instantaneous batch size per device =" f" {per_device_train_batch_size}")
    logger.info("  Gradient accumulation steps =" f" {gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size * gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")


    # ======================== Training ================================
    train_time = 0
    train_start = time.time()
    steps_trained_progress_bar = tqdm(
        range(total_train_steps), desc="Train steps ... ", position=0, disable=not accelerator.is_local_main_process
    )
    continue_training = True
    epochs_trained = 0
    cur_step = 0

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        if training_args.push_to_hub:
            api = HfApi(token=training_args.hub_token)

            # Create repo (repo_name from args or inferred)
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "wandb" not in gitignore:
                    gitignore.write("wandb\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    # only the main process saves them
    if accelerator.is_main_process:
        # save feature extractor, tokenizer and config
        if (
            model_args.prompt_tokenizer_name is None
            and model_args.description_tokenizer_name
            or (model_args.prompt_tokenizer_name == model_args.description_tokenizer_name)
        ):
            prompt_tokenizer.save_pretrained(training_args.output_dir)
        else:
            logger.warning(
                f"Prompt tokenizer ('{model_args.prompt_tokenizer_name}') and description tokenizer ('{model_args.description_tokenizer_name}') are not the same. Saving only the prompt tokenizer."
            )
            prompt_tokenizer.save_pretrained(training_args.output_dir)

        feature_extractor.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()

    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        # Find num steps and epoch from saved state string pattern
        pattern = r"checkpoint-(\d+)-epoch-(\d+)"
        match = re.search(pattern, checkpoint)
        cur_step = int(match.group(1))
        epochs_trained = int(match.group(2))

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"  Continuing training from epoch {epochs_trained}")
        logger.info(f"  Continuing training from global step {cur_step}")

        steps_trained_progress_bar.update(cur_step)

        for epoch in range(0, epochs_trained):
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)

        if training_args.max_steps < 0:
            # we know exactly the number of steps per epoch, so can skip through the required number of batches
            resume_step = (cur_step - epochs_trained * steps_per_epoch) * gradient_accumulation_steps
        else:
            # Currently we don't know how many steps we've taken in the current epoch
            # So we just shuffle the dataset one extra time and start from a fresh epoch
            # This is "good enough" for our purposes but not fully correct
            resume_step = None
            with accelerator.local_main_process_first():
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
    else:
        resume_step = None

    gen_kwargs = {
        "do_sample": model_args.do_sample,
        "temperature": model_args.temperature,
        "max_length": model_args.max_length,
        # Because of the delayed pattern mask, generation might stop earlier because of unexpected behaviour
        # on the first tokens of the codebooks that are delayed.
        # This fix the issue.
        "min_new_tokens": num_codebooks + 1,
    }

    def dynamic_loss_weights_step(step, total_steps, num_codebooks=8, initial_weights=None):
        step_boundaries = [0.3, 0.6, 0.8] 
        alpha_values = [0.0, 0.3, 0.6, 0.8] 

        current_stage = torch.tensor([step / total_steps]).float() 
        current_stage = torch.bucketize(current_stage, torch.tensor(step_boundaries)) 
        alpha = torch.tensor(alpha_values)[current_stage] 

        if initial_weights is None:   
            initial_weights = torch.tensor([0.25, 0.25, 0.25] + [0.05] * (num_codebooks - 3)).float()
        else:
            initial_weights = torch.tensor(initial_weights).float()

        final_weights = torch.ones(num_codebooks).float() / num_codebooks

        weights = (1 - alpha) * initial_weights + alpha * final_weights
        return weights

    # Define gradient update step fn
    def train_step(
        batch, 
        accelerator,
        cur_step,
        total_train_step,
        autocast_kwargs,
        weight_loss=True,
        add_vocal_loss=False,
        num_codebook=8,
    ):
        if mixed_precision == "fp16":
            # fp16 doesn't work with T5-like models
            with accelerator.autocast(autocast_handler=autocast_kwargs):
                if training_args.parallel_mode.value != "distributed":
                    encoder_outputs = model.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                else:
                    encoder_outputs = model.module.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                # we optionnally project last_hidden_state to avoid recomputing every time
                encoder_hidden_states = encoder_outputs.last_hidden_state
                if (
                    config.text_encoder.hidden_size != config.decoder.hidden_size
                    and config.decoder.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = (
                        model.enc_to_dec_proj(encoder_hidden_states)
                        if training_args.parallel_mode.value != "distributed"
                        else model.module.enc_to_dec_proj(encoder_hidden_states)
                    )

                if batch.get("attention_mask", None) is not None:
                    encoder_hidden_states = encoder_hidden_states * batch.get("attention_mask", None)[..., None]

                encoder_outputs.last_hidden_state = encoder_hidden_states
                batch["encoder_outputs"] = encoder_outputs

        outputs = model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss.item()}

        if weight_loss:
            codebook_weights = dynamic_loss_weights_step(cur_step, total_train_step)
            assert len(codebook_weights)== num_codebook, f'len(codebook_weight): {len(codebook_weights)}, codebook:{len(codebook_weight)}'
        if outputs.get("codebook_losses", None) is not None:
            total_codebook_loss = 0
            acc_codebook_loss = 0
            vocal_codebook_loss = 0

            for i, loss in enumerate(outputs.codebook_losses): 
                metrics[f'code_loss_{i}'] = loss.item()  
                if weight_loss:
                    if i < num_codebook:
                        acc_codebook_loss += codebook_weights[i % num_codebook] * loss
                    else:
                        vocal_codebook_loss += codebook_weights[i % num_codebook] * loss
                    metrics[f'loss_weight_{i}'] = codebook_weights[i % num_codebook] 
                else:
                    if i < num_codebook:
                        acc_codebook_loss += loss/num_codebook
                    else:
                        vocal_codebook_loss += loss/num_codebook
            if len(outputs.codebook_losses) > num_codebook:
                total_codebook_loss = 0.5 * acc_codebook_loss + 0.5 * vocal_codebook_loss #0.4 * acc_codebook_loss + 0.6 * vocal_codebook_loss
                metrics[f'acc_head_weighted_loss'] = acc_codebook_loss.item()
                metrics[f'vocal_head_weighted_loss'] = vocal_codebook_loss.item()
            else:
                assert vocal_codebook_loss == 0, f'num lm_heads = {num_codebook}, but vocal_codebook_loss = {vocal_codebook_loss}'
                total_codebook_loss = acc_codebook_loss
            if weight_loss:
                ce_loss = total_codebook_loss
                metrics["weighted_loss"]= total_codebook_loss.item()
    
        if add_vocal_loss: 
            metrics["vocal_loss"]=outputs.vocal_loss.item()
            vocal_codebook_weights = dynamic_loss_weights_step(cur_step, total_train_step)
            vocal_total_codebook_loss =0
            for i, loss in enumerate(outputs.vocal_codebook_losses): 
                metrics[f'vocal_code_loss_{i}'] = loss.item()  
                vocal_total_codebook_loss += vocal_codebook_weights[i] * loss
                metrics[f'vocal_loss_weight_{i}'] = vocal_codebook_weights[i] 
            
            ce_loss += 0.1 * vocal_total_codebook_loss
            metrics["vocal_weighted_loss"]= vocal_total_codebook_loss.item() 

        return ce_loss, metrics

    

    # Define eval fn
    def eval_step(
        batch,
        accelerator,
        autocast_kwargs,
    ):
        eval_model = model if not training_args.torch_compile else model._orig_mod

        if mixed_precision == "fp16":
            # fp16 doesn't work with T5-like models
            with accelerator.autocast(autocast_handler=autocast_kwargs):
                if training_args.parallel_mode.value != "distributed":
                    encoder_outputs = model.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                else:
                    encoder_outputs = model.module.text_encoder(
                        input_ids=batch.get("input_ids"), attention_mask=batch.get("attention_mask", None)
                    )
                # we optionnally project last_hidden_state to avoid recomputing every time
                encoder_hidden_states = encoder_outputs.last_hidden_state
                if (
                    config.text_encoder.hidden_size != config.decoder.hidden_size
                    and config.decoder.cross_attention_hidden_size is None
                ):
                    encoder_hidden_states = (
                        model.enc_to_dec_proj(encoder_hidden_states)
                        if training_args.parallel_mode.value != "distributed"
                        else model.module.enc_to_dec_proj(encoder_hidden_states)
                    )

                if batch.get("attention_mask", None) is not None:
                    encoder_hidden_states = encoder_hidden_states * batch.get("attention_mask", None)[..., None]

                encoder_outputs.last_hidden_state = encoder_hidden_states
                batch["encoder_outputs"] = encoder_outputs

        with torch.no_grad():
            outputs = eval_model(**batch)
        # CE (data) loss
        ce_loss = outputs.loss
        metrics = {"loss": ce_loss}
        if outputs.get("codebook_losses", None) is not None:
            for i, loss in enumerate(outputs.codebook_losses): 
                metrics[f'code_loss_{i}'] = loss 
        
        if outputs.get("vocal_codebook_losses", None) is not None:
            for i, loss in enumerate(outputs.vocal_codebook_losses): 
                metrics[f'vocal_code_loss_{i}'] = loss 
        if outputs.get("vocal_loss", None) is not None:
            metrics[f'vocal_loss'] = outputs.vocal_loss 

        return metrics

    def generate_step(batch, accelerator):
        batch.pop("decoder_attention_mask", None)
        eval_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
        if training_args.torch_compile:
            # if the model is compiled, we use the original model bc compile is not compatible with .generate
            eval_model = model._orig_mod

        # since we've might have loaded the weights in fp32, we have to autocast to ensure FA2 weights are in half-precision.
        # with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=(attn_implementation=="flash_attention_2"))):
        acc_output_audios, vocal_output_audios = eval_model.generate(**batch, **gen_kwargs)
        acc_output_audios = accelerator.pad_across_processes(acc_output_audios, dim=1, pad_index=0)
        vocal_output_audios = accelerator.pad_across_processes(vocal_output_audios, dim=1, pad_index=0)
        return acc_output_audios, vocal_output_audios
    

    def decode_codes_audios_AV(labels, accelerator, track_pattern, n_codebooks = 8):
        # shape: (batch_size, seq_len, num_codebooks) -> (batch_size, num_codebooks, time_steps)
        # audio_decoder  shape: (batch_size, num_codebooks, time_steps)
        eval_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
        if training_args.torch_compile:
            audio_decoder =  eval_model._orig_mod.audio_encoder 
        else:
            audio_decoder = eval_model.audio_encoder
        label_audios = []
        output_lengths = []
        audio_decoder.to(labels.device).eval()  
        labels = labels.transpose(2, 1)
        for label in labels:
            mask = (label >= 0) & (label < 1024)

            # assert audio_decoder.model.codebook_size == 1024, f'Expected codebook_size of 1024, but got {audio_decoder.codebook_size}'
            # assert audio_decoder.model.n_codebooks == 8
            label = label[mask]

            if track_pattern.startswith('parallel'):
                combined_num_codebooks = n_codebooks * 2
            else:
                combined_num_codebooks = n_codebooks

            label = label.reshape(1, combined_num_codebooks, -1)
            acc_label, vocal_label = split_combined_track_input_ids(label, track_pattern, n_codebooks)
            
            acc_label = acc_label.reshape(1, 1, n_codebooks, -1)
            vocal_label = vocal_label.reshape(1, 1, n_codebooks, -1)
            assert acc_label.shape == vocal_label.shape, f"acc_label.shape: {acc_label.shape}, vocal_label.shape:{vocal_label.shape} "
            # print(f"Mask applied, remaining shape: {label.shape}")

            with torch.no_grad():
                acc_audio = audio_decoder.decode(acc_label, [None]).audio_values  # 解码为音频 (batch_size, input_length)
                vocal_audio = audio_decoder.decode(vocal_label, [None]).audio_values
            audio = acc_audio + vocal_audio 
            audio = audio.transpose(0, 2) #(seq_len, 1, 1)
            label_audios.append(audio)
            output_lengths.append(audio.shape[0])
        output_values = (
                torch.nn.utils.rnn.pad_sequence(label_audios, batch_first=True, padding_value=0)
                .squeeze(-1)
                .squeeze(-1)
            )
        output_values = accelerator.pad_across_processes(output_values, dim=1, pad_index=0)
        output_lengths = torch.tensor(output_lengths).to(output_values.device)

        return {'output_values': output_values, 'output_lengths': output_lengths}

        
        
    model.train()
    log_gt_first_flag = True
    for epoch in range(epochs_trained, num_epochs):
        with accelerator.local_main_process_first():
            vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(training_args.seed)
        sampler = None
        if training_args.group_by_length:
            sampler = LengthGroupedSampler(train_batch_size, lengths=vectorized_datasets["train"]["target_length"])
        train_dataloader = DataLoader(
            vectorized_datasets["train"],
            collate_fn=data_collator,
            batch_size=per_device_train_batch_size,
            sampler=sampler,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        if hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(epoch)

        if resume_step is not None:
            # Skip the first N batches in the dataloader when resuming from a checkpoint
            logger.info(f"  Skip first {resume_step} batches")
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            resume_step = None
            accelerator.wait_for_everyone()

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss, train_metric = train_step(batch, accelerator, cur_step, total_train_steps, autocast_kwargs, weight_loss=training_args.codebook_weighting, add_vocal_loss=training_args.add_vocal_loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                steps_trained_progress_bar.update(1)
                cur_step += 1

                if cur_step % training_args.logging_steps == 0:
                    if 'vocal_loss' in train_metric:
                        steps_trained_progress_bar.write(
                        f"Step... ({cur_step} / {total_train_steps} | Loss:"
                        f" {train_metric['loss']}, Vocal Loss: {train_metric['vocal_loss']} Learning Rate:"
                        f" {lr_scheduler.get_last_lr()[0]})"
                    )
                    else:
                        steps_trained_progress_bar.write(
                            f"Step... ({cur_step} / {total_train_steps} | Loss:"
                            f" {train_metric['loss']}, Learning Rate:"
                            f" {lr_scheduler.get_last_lr()[0]})"
                        )
                    log_metric(
                        accelerator,
                        metrics=train_metric,
                        learning_rate=lr_scheduler.get_last_lr()[0],
                        train_time=train_time + time.time() - train_start,
                        step=cur_step,
                        epoch=epoch,
                        prefix="train",
                    )

                # save checkpoint and weights after each save_steps and at the end of training
                if (cur_step % training_args.save_steps == 0) or cur_step == total_train_steps:
                    intermediate_dir = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}-epoch-{epoch}")
                    # safe_serialization=False to avoid shared tensors saving issue (TODO(YL): it's a temporary fix)
                    # https://github.com/huggingface/transformers/issues/27293#issuecomment-1872560074
                    accelerator.save_state(output_dir=intermediate_dir, safe_serialization=False)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(
                            training_args.save_total_limit, output_dir=training_args.output_dir, logger=logger
                        )

                        if cur_step == total_train_steps:
                            # un-wrap student model for save
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(training_args.output_dir)

                        if training_args.push_to_hub:
                            api.upload_folder(
                                repo_id=repo_id,
                                folder_path=training_args.output_dir,
                                commit_message=f"Saving train state of step {cur_step}",
                                run_as_future=True,
                            )
                    accelerator.wait_for_everyone()

                if training_args.do_eval and (cur_step % eval_steps == 0 or cur_step == total_train_steps):
                    train_time += time.time() - train_start
                    # ======================== Evaluating ==============================
                    model.eval()
                    eval_metrics = []
                    eval_preds_acc = []
                    eval_preds_vocal = []
                    eval_descriptions = []
                    eval_prompts = []
                    eval_start = time.time()

                    if log_gt_first_flag:
                        gt_refvoices = []
                        gt_labels = []
                        gt_label_lens = []
                        gt_refaudios = []
                        gt_refaudio_lens = []
                        
                    else:
                        gt_refvoices = None
                        gt_labels = None
                        gt_refaudios = None


                    # release training input batch
                    batch = release_memory(batch)

                    validation_dataloader = DataLoader(
                        vectorized_datasets["eval"],
                        collate_fn=data_collator,
                        batch_size=per_device_eval_batch_size,
                        drop_last=False,
                        num_workers=training_args.eval_dataloader_num_workers,
                        pin_memory=training_args.dataloader_pin_memory,
                    )
                    validation_dataloader = accelerator.prepare(validation_dataloader)

                    for batch in tqdm(
                        validation_dataloader,
                        desc=f"Evaluating - Inference ...",
                        position=2,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Model forward
                        # pdb.set_trace()
                        eval_metric = eval_step(batch, accelerator, autocast_kwargs)
                        eval_metric = accelerator.gather_for_metrics(eval_metric)
                        eval_metric = {key: val.unsqueeze(0) if val.ndim == 0 else val for (key,val) in eval_metric.items()}
                        eval_metrics.append(eval_metric)

                    if training_args.predict_with_generate:
                        validation_dataloader = DataLoader(
                            vectorized_datasets["eval"],
                            collate_fn=data_collator,
                            batch_size=per_device_eval_batch_size,
                            drop_last=False,
                            num_workers=training_args.dataloader_pin_memory,
                            pin_memory=training_args.dataloader_pin_memory,
                        )
                        validation_dataloader = accelerator.prepare(validation_dataloader)
                        # generation
                        for batch in tqdm(
                            validation_dataloader,
                            desc=f"Evaluating - Generation ...",
                            position=2,
                            disable=not accelerator.is_local_main_process,
                        ):
                            acc_generated_audios, vocal_generated_audios = generate_step(batch, accelerator)
                            # Gather all predictions and targets
                            acc_generated_audios, vocal_generated_audios, input_ids, prompts  = accelerator.pad_across_processes(
                                (acc_generated_audios, vocal_generated_audios, batch["input_ids"], batch["prompt_input_ids"]), dim=1, pad_index=0
                            )
                            acc_generated_audios, vocal_generated_audios, input_ids, prompts = accelerator.gather_for_metrics(
                                (acc_generated_audios, vocal_generated_audios, input_ids, prompts)
                            )

                            eval_preds_acc.extend(acc_generated_audios.to("cpu"))
                            eval_preds_vocal.extend(vocal_generated_audios.to("cpu"))
                            eval_descriptions.extend(input_ids.to("cpu"))
                            eval_prompts.extend(prompts.to("cpu"))

                            logger.warning(f"rank: {training_args.local_rank}, log_gt_first_flag: {log_gt_first_flag}")
                            if log_gt_first_flag:
                                #decode labels:
                                logger.warning(f"rank: {training_args.local_rank}, log ground truth")
                                label_audios_dict = decode_codes_audios_AV(batch['labels'], accelerator, model_args.track_pattern)

                                label_audios_dict= accelerator.pad_across_processes(
                                    label_audios_dict, dim=1, pad_index=0
                                )
                                label_audios_dict = accelerator.gather_for_metrics(
                                    label_audios_dict
                                )
                                gt_labels.extend(label_audios_dict['output_values'].to("cpu"))
                                gt_label_lens.extend(label_audios_dict['output_lengths'].to('cpu'))
                                

                                if "ref_audio_ids" in batch:
                                    ref_audios_dict = decode_codes_audios_AV(batch['ref_audio_ids'], accelerator, model_args.track_pattern)
                                    ref_audios_dict= accelerator.pad_across_processes( ref_audios_dict, dim=1, pad_index=0 )
                                    ref_audios_dict = accelerator.gather_for_metrics( ref_audios_dict  )
                                    gt_refaudios.extend(ref_audios_dict['output_values'].to("cpu"))
                                    gt_refaudio_lens.extend(ref_audios_dict['output_lengths'].to('cpu'))


                                if "ref_voice_values" in batch:
                                    ref_voices = batch["ref_voice_values"]
                                    ref_voices = accelerator.pad_across_processes(
                                        ref_voices, dim=1, pad_index=0
                                    )
                                    ref_voices = accelerator.gather_for_metrics( ref_voices )
                                    gt_refvoices.extend(ref_voices.to("cpu"))
                                

                    eval_time = time.time() - eval_start



                    # normalize eval metrics
                    eval_metrics = {
                        key: torch.mean(torch.cat([d[key] for d in eval_metrics])).to("cpu") for key in eval_metrics[0]
                    }

                    # compute metrics
                    metrics_desc = ""
                    if training_args.predict_with_generate:
                        if accelerator.is_local_main_process:
                            (
                                metric_values,
                                pred_descriptions,
                                pred_prompts,
                                acc_audios,
                                vocal_audios,
                                audios,
                                transcriptions,
                                si_sdr_measures,
                            ) = compute_metrics(
                                eval_preds_acc,
                                eval_preds_vocal,
                                eval_descriptions,
                                eval_prompts,
                                accelerator.device,
                                training_args.compute_clap_similarity_metric,
                                training_args.compute_noise_level_metric,
                                training_args.noise_level_to_compute_clean_wer,
                            )
                            eval_metrics.update(metric_values)
                            metrics_desc = " ".join([f"Eval {key}: {value} |" for key, value in metric_values.items()])
                            if "wandb" in training_args.report_to:
                                log_pred_AV(
                                    accelerator,
                                    pred_descriptions,
                                    pred_prompts,
                                    transcriptions,
                                    acc_audios,
                                    vocal_audios,
                                    audios,
                                    si_sdr_measures,
                                    sampling_rate=sampling_rate,
                                    step=cur_step,
                                    prefix="eval",
                                )

                                logger.warning(f"rank: {training_args.local_rank}, log_gt_first_flag: {log_gt_first_flag}, log_gt()")
                                if log_gt_first_flag:
                                    log_gt(
                                        accelerator,
                                        pred_descriptions,
                                        pred_prompts,
                                        gt_labels=[a.float().cpu().numpy() for a in gt_labels],
                                        gt_label_lens=gt_label_lens,                                    
                                        sampling_rate=sampling_rate,
                                        gt_refaudios=[a.float().cpu().numpy() for a in gt_refaudios] if gt_refaudios else None,
                                        gt_refaudio_lens=gt_refaudio_lens if gt_refaudio_lens else None ,       
                                        gt_refvoices=[a.float().cpu().numpy() for a in gt_refvoices] if gt_refvoices else None,
                                        refvoice_sr= refvoice_sr if ref_voice_column_name is not None else 0,
                                        step=cur_step,
                                    )
                                    
                                    gt_labels, gt_refaudios, gt_refvoices, gt_label_lens, gt_refaudio_lens = release_memory(gt_labels, gt_refaudios, gt_refvoices, gt_label_lens, gt_refaudio_lens)
                                    
                        accelerator.wait_for_everyone()

                    # Print metrics and update progress bar
                    if accelerator.is_local_main_process:
                        steps_trained_progress_bar.write(
                            f"Eval results for step ({cur_step} / {total_train_steps} | Eval Loss: {eval_metrics['loss']} |"
                            f" {metrics_desc})"
                        )

                    log_metric(
                        accelerator,
                        metrics=eval_metrics,
                        train_time=eval_time,
                        step=cur_step,
                        epoch=epoch,
                        prefix="eval",
                    )

                    # release eval batch and relax metrics
                    eval_metrics, eval_preds_acc, eval_preds_vocal, eval_descriptions, eval_prompts,  batch, eval_metric = release_memory(
                        eval_metrics, eval_preds_acc, eval_preds_vocal, eval_descriptions, eval_prompts,  batch, eval_metric
                    )
                    if training_args.predict_with_generate:
                        acc_generated_audios, vocal_generated_audios, input_ids, prompts = release_memory(acc_generated_audios, vocal_generated_audios, input_ids, prompts)

                    # train mode
                    model.train()

                    # flush the train metrics
                    train_start = time.time()

                # break condition
                if cur_step == total_train_steps:
                    continue_training = False
                    break

        if not continue_training:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
