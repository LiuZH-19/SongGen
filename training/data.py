import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
import numpy as np
import torch
from accelerate import Accelerator
import datasets
from datasets import Dataset, IterableDataset, concatenate_datasets, interleave_datasets, load_dataset, DownloadMode
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer
import os
from transformers import Wav2Vec2FeatureExtractor
import torchaudio.transforms as T
import random
from songgen import (
    VoiceBpeTokenizer,
    combine_track_input_ids
)
from .my_proxy import *


@dataclass
class DataCollatorCodecWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch or
    to `max_length` if `max_length` is set and `padding=max_length`.
    """

    feature_extractor: AutoFeatureExtractor
    audio_column_name: str
    ref_audio_column_name: Optional[list] = None
    feature_extractor_input_name: Optional[str] = "input_values"
    max_length: Optional[int] = None
    padding: Optional[str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        audios = [feature[self.audio_column_name]["array"] for feature in features]
        len_audio = [len(audio) for audio in audios]
        if self.max_length is not None:
            audios = [audio[: min(l, self.max_length)] for audio, l in zip(audios, len_audio)]

        # since resampling has already been performed in the 'load_multiple_datasets' function,
        # a fixed sampling_rate(16khz) is passed to the feature_extractor.
        sampling_rate = self.feature_extractor.sampling_rate
        batch = self.feature_extractor(
            audios, sampling_rate=sampling_rate, return_tensors="pt", padding=self.padding, max_length=self.max_length
        )
        batch["len_audio"] = torch.tensor(len_audio).unsqueeze(1)

        if self.ref_audio_column_name:
            refbatch_ALL ={}
            
            for ref_col in self.ref_audio_column_name:
                atype = ref_col.split('_')[0]
                refaudios = []
                for feature in features:
                    wav = feature[ref_col]["array"]
                    if wav is None:
                        # If ref_audio is empty, create an all-zero array of the same length as audio_column
                        wav = np.zeros_like(feature[self.audio_column_name]["array"])
                        
                    refaudios.append(wav)
                ref_len_audio = [len(audio) for audio in refaudios]
                if self.max_length is not None:
                    refaudios = [audio[: min(l, self.max_length)] for audio, l in zip(refaudios, ref_len_audio)]

                refbatch = self.feature_extractor(
                    refaudios, sampling_rate=sampling_rate, return_tensors="pt", padding=self.padding, max_length=self.max_length
                )
                refbatch["len_audio"] = torch.tensor(ref_len_audio).unsqueeze(1)
                refbatch_ALL[atype] = refbatch

        return (batch , refbatch_ALL)




# convert t5 tokenizer to voicebpetokenizer
@dataclass
class DataCollatorSongGenWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        prompt_tokenizer (:class:`~transformers.AutoTokenizer`)
            The prompt_tokenizer used for proccessing the data.
        description_tokenizer (:class:`~transformers.AutoTokenizer`)
            The description_tokenizer used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    voicebpe_tokenizer: VoiceBpeTokenizer
    prompt_tokenizer: AutoTokenizer
    description_tokenizer: AutoTokenizer
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    prompt_max_length: Optional[int] = None
    description_max_length: Optional[int] = None
    audio_max_length: Optional[int] = None
    mert_processor: Optional[Wav2Vec2FeatureExtractor] = None
    ref_dur_sec: Optional[int] = None
    ref_voice_column_name: Optional[str] = None
    ref_audio_column_name: Optional[Union[List[str], str]] = None
    audio_encoder_bos_token_id: Optional[int] = None
    audio_encoder_eos_token_id: Optional[int] = None
    audio_encoder_special_token_ids: Optional[dict] = None
    id_column_name: Optional[str] = None
    label_atype: Optional[str] ="labels"   #Possible values for "label_atype" include: "labels" (mixed audio), "acc+vocal" (dual-track mode), or individual tracks such as "acc" (accompaniment) or "vocal" (vocals).
    add_vocal_labels: Optional[bool] = False
    energy_name: Optional[str] = '_clip_energy'
    track_pattern: Optional[str] = None
    num_codebooks: Optional[int] = 8
    

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        if self.label_atype =='labels':
            labels = [torch.tensor(feature["labels"]).transpose(0, 1) for feature in features]
        
        elif self.label_atype == 'acc+vocal' and self.track_pattern is not None: 
            labels = []
            for feature in features:
                acc_label = torch.tensor(feature[f"ref_audio_ids_acc"])[:, 1:] #(num_codebook, seq_len) -> (seq_len, num_codebook)
                acc_label.masked_fill_(acc_label==self.audio_encoder_special_token_ids['acc']['bos'], self.audio_encoder_bos_token_id)
                acc_label.masked_fill_(acc_label==self.audio_encoder_special_token_ids['acc']['eos'],  self.audio_encoder_eos_token_id)

                vocal_label = torch.tensor(feature[f"ref_audio_ids_vocal"])[:, 1:]
                vocal_label.masked_fill_(vocal_label==self.audio_encoder_special_token_ids['vocal']['bos'], self.audio_encoder_bos_token_id)
                vocal_label.masked_fill_(vocal_label==self.audio_encoder_special_token_ids['vocal']['eos'],  self.audio_encoder_eos_token_id)
                combined_label = combine_track_input_ids(acc_label, vocal_label, self.track_pattern, self.audio_encoder_bos_token_id, self.audio_encoder_eos_token_id, self.num_codebooks)
                combined_label = combined_label.reshape(-1, combined_label.shape[-1]).transpose(0, 1)
                labels.append(combined_label)   

        else: #When label_atype is set to another track, such as vocal.
            labels = []
            for feature in features:
                if isinstance(self.label_atype , list):
                        label_atype = random.choice(self.label_atype) 
                        if label_atype != 'labels' and  feature.get(label_atype + self.energy_name, 3000) < 2000:
                            label_atype ='labels'
                else:
                    label_atype = self.label_atype
                if label_atype =='labels':
                    tmp = torch.tensor(feature["labels"]).transpose(0, 1)
                else:
                    tmp = torch.tensor(feature[f"ref_audio_ids_{label_atype}"])[:, 1:].transpose(0, 1)
                    tmp.masked_fill_(tmp==self.audio_encoder_special_token_ids[label_atype]['bos'], self.audio_encoder_bos_token_id)
                    tmp.masked_fill_(tmp==self.audio_encoder_special_token_ids[label_atype]['eos'],  self.audio_encoder_eos_token_id)
                labels.append(tmp)
        
        # (bsz, seq_len, num_codebooks)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        if self.add_vocal_labels:
            vocal_labels = []
            for feature in features:
                tmp = torch.tensor(feature[f"ref_audio_ids_vocal"])[:, 1:].transpose(0, 1)
                tmp.masked_fill_(tmp==self.audio_encoder_special_token_ids['vocal']['bos'], self.audio_encoder_bos_token_id)
                tmp.masked_fill_(tmp==self.audio_encoder_special_token_ids['vocal']['eos'],  self.audio_encoder_eos_token_id)
                vocal_labels.append(tmp)
            # (bsz, seq_len, num_codebooks)
            vocal_labels = torch.nn.utils.rnn.pad_sequence(vocal_labels, batch_first=True, padding_value=-100)

            if vocal_labels.shape  != labels.shape:
                print(f'to Convert In DataCollator vocal_labels: {vocal_labels.shape}, labels: {labels.shape}')
                if vocal_labels.shape[1] < labels.shape[1]:
                    print(f'*<* to Convert In DataCollator vocal_labels: {vocal_labels.shape}, labels: {labels.shape}')
                    vocal_labels = torch.nn.functional.pad(
                        vocal_labels, pad=(0, 0, 0, max(labels.shape[1] - vocal_labels.shape[1], 0)), value=-100
                    )
                else:
                    print(f'*>* to Convert In DataCollator vocal_labels: {vocal_labels.shape}, labels: {labels.shape}')
                    vocal_labels = vocal_labels[:, : labels.shape[1],:]
            assert vocal_labels.shape == labels.shape, f'In DataCollator vocal_labels: {vocal_labels.shape}, labels: {labels.shape}'

        if self.audio_max_length is not None and self.padding == "max_length":
            labels = torch.nn.functional.pad(
                labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)), value=-100
            )

            if self.add_vocal_labels:
                vocal_labels = torch.nn.functional.pad(
                    vocal_labels, pad=(0, 0, 0, max(self.audio_max_length - labels.shape[1], 0)), value=-100
                )
        
        batch = {"labels": labels}
        if self.add_vocal_labels :
            batch['vocal_labels'] = vocal_labels

        if "input_ids" in features[0].keys(): 
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            input_ids = self.description_tokenizer.pad(
                input_ids,
                return_tensors="pt",
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.description_max_length,
            )
            
            batch.update(input_ids)
       

        if "prompt_input_ids" in features[0].keys(): 
            prompt_input_ids = [{"input_ids": feature["prompt_input_ids"]} for feature in features]
            prompt_input_ids = self.prompt_tokenizer.pad( 
                prompt_input_ids,
                return_tensors="pt",
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.prompt_max_length,
            )

            batch["prompt_input_ids"] = prompt_input_ids["input_ids"]
            if "attention_mask" in prompt_input_ids:
                batch["prompt_attention_mask"] = prompt_input_ids["attention_mask"]
        

        if self.ref_voice_column_name is not None:
            if "ref_voice_values" in features[0].keys() and "mert_input" in features[0].keys():
                logger.warning_once("Both 'ref_voice_values' and 'mert_input' are in dataset")
            elif self.ref_voice_column_name!= 'random' and "ref_voice_values" in features[0].keys(): 
                ref_voice_values = torch.cat([torch.tensor(feature["ref_voice_values"]) for feature in features], dim=0)
                ref_voice_attention_mask = torch.cat([torch.tensor(feature["ref_voice_attention_mask"]) for feature in features], dim=0)
                batch["ref_voice_values"] = ref_voice_values
                batch["ref_voice_attention_mask"] = ref_voice_attention_mask
            elif self.ref_voice_column_name!= 'random' and "mert_input" in features[0].keys():
                input_wavs = []
                for feature in features:
                    if  feature.get('in_vad', 1)== 0: 
                        input_wavs.append(np.zeros((self.ref_dur_sec*self.mert_processor.sampling_rate,)))
                        continue
                    if isinstance(self.ref_voice_column_name , list):
                        voice_type = random.choice(self.ref_voice_column_name)
                        if voice_type == 'random' or feature.get('vocal'+self.energy_name, 300) < 200 or feature.get('in_vad', 1) == 0:
                            input_wavs.append(np.zeros((self.ref_dur_sec*self.mert_processor.sampling_rate,)))
                            continue
                    
                    audio = feature['mert_input']
                    sr = audio["sampling_rate"]
                    wav = audio["array"]
                    if sr < 0: #Handle the case when ref_voice is None
                        sr = self.mert_processor.sampling_rate
                        input_wav = np.zeros((self.ref_dur_sec*self.mert_processor.sampling_rate,))
                    else:
                        assert sr == self.mert_processor.sampling_rate,  f"sr {sr} != mert_processor.sampling_rate: {self.mert_processor.sampling_rate}"
                        ref_dur = int(self.ref_dur_sec * sr)
                        if len(wav) > ref_dur + 1:
                            lidx = np.random.randint(low=0, high=len(wav) - ref_dur - 1)
                            ridx = lidx + ref_dur
                            input_wav = wav[lidx:ridx]
                        else:
                            input_wav= wav
                    input_wavs.append(input_wav)
                mert_inputs = self.mert_processor(input_wavs, sampling_rate=self.mert_processor.sampling_rate, return_tensors="pt", padding="max_length", max_length=self.ref_dur_sec*self.mert_processor.sampling_rate)
                assert mert_inputs['input_values'].shape[-1] == self.ref_dur_sec*self.mert_processor.sampling_rate, f"ref_voice_values.shape:  {mert_inputs['input_values'].shape}"
                batch['ref_voice_values'] = mert_inputs['input_values']
                batch['ref_voice_attention_mask'] = mert_inputs['attention_mask']
            elif self.ref_voice_column_name == 'random' and self.mert_processor is not None and  self.ref_dur_sec is not None: 
                input_wavs= [np.zeros((self.ref_dur_sec*self.mert_processor.sampling_rate,)) for feature in features] 
                mert_inputs = self.mert_processor(input_wavs, sampling_rate=self.mert_processor.sampling_rate, return_tensors="pt", padding="max_length", max_length=self.ref_dur_sec*self.mert_processor.sampling_rate)
                assert mert_inputs['input_values'].shape[-1] == self.ref_dur_sec*self.mert_processor.sampling_rate, f"ref_voice_values.shape:  {mert_inputs['input_values'].shape}"
                batch['ref_voice_values'] = mert_inputs['input_values']
                batch['ref_voice_attention_mask'] = mert_inputs['attention_mask']

        
        #ref_audio_ids can be constructed by randomly selecting one or any combination among acc, vocal, and drum, based on ref_audio_column_name.
        if self.ref_audio_column_name is not None:
            if isinstance(self.ref_audio_column_name, str): 
                if "ref_audio_ids" in features[0].keys() and self.ref_audio_column_name != 'random':
                    ref_audio_ids = [torch.tensor(feature["ref_audio_ids"]).transpose(0, 1) for feature in features]
                    # (bsz, seq_len, num_codebooks)
                    ref_audio_ids = torch.nn.utils.rnn.pad_sequence(ref_audio_ids, batch_first=True, padding_value=-100)
                    if self.audio_max_length is not None and self.padding == "max_length":
                        ref_audio_ids = torch.nn.functional.pad(
                            ref_audio_ids, pad=(0, 0, 0, max(self.audio_max_length - ref_audio_ids.shape[1], 0)), value=-100
                        )
                    batch["ref_audio_ids"] = ref_audio_ids
                else:
                    bos_labels = torch.ones((labels.shape[0], 1 , 9), dtype=torch.long) * self.audio_encoder_bos_token_id
                    ref_audio_ids = torch.cat([bos_labels, labels.clone()], dim=1).long() 
                    ref_audio_ids.masked_fill_(ref_audio_ids == self.audio_encoder_bos_token_id, self.audio_encoder_special_token_ids['random']['bos'])
                    ref_audio_ids.masked_fill_(ref_audio_ids == self.audio_encoder_eos_token_id, self.audio_encoder_special_token_ids['random']['eos'])
                    codec_mask = (ref_audio_ids >= 0) & (ref_audio_ids < 1024)
                    ref_audio_ids.masked_fill_(codec_mask, self.audio_encoder_special_token_ids['mask'])
                    batch["ref_audio_ids"] = ref_audio_ids
            elif isinstance(self.ref_audio_column_name, list):
                ref_audio_ids = []
                for feature in features:
                    valid_candidates = []
                    energy_bar = 200 #2000
                    for atype in self.ref_audio_column_name:
                        if atype == 'random':
                            valid_candidates.append('random')
                        else:
                            if atype in ['acc', 'drum'] and feature.get(atype + self.energy_name, energy_bar) >= energy_bar:
                                valid_candidates.append(atype)
                            elif atype in ['vocal'] and feature.get(atype + self.energy_name, energy_bar) >= energy_bar and len(feature['prompt_input_ids']) > 1:
                                valid_candidates.append(atype)
                            
                    if not valid_candidates:
                        select_atype = 'random'  
                    else:
                        select_atype = random.choice(valid_candidates)
                    if select_atype == 'random': 
                        item_ids = torch.tensor(feature[f"ref_audio_ids_acc"]).transpose(0, 1)  
                        item_ids.masked_fill_(item_ids == self.audio_encoder_special_token_ids['acc']['bos'], self.audio_encoder_special_token_ids['random']['bos'])
                        item_ids.masked_fill_(item_ids == self.audio_encoder_special_token_ids['acc']['eos'], self.audio_encoder_special_token_ids['random']['eos'])
                        codec_mask = (item_ids >= 0) & (item_ids < 1024)
                        item_ids.masked_fill_(codec_mask, self.audio_encoder_special_token_ids['mask'])
                    else:
                        item_ids = torch.tensor(feature[f"ref_audio_ids_{select_atype}"]).transpose(0, 1)
                    ref_audio_ids.append(item_ids)
                ref_audio_ids = torch.nn.utils.rnn.pad_sequence(ref_audio_ids, batch_first=True, padding_value=-100)
                if self.audio_max_length is not None and self.padding == "max_length":
                    ref_audio_ids = torch.nn.functional.pad(
                        ref_audio_ids, pad=(0, 0, 0, max(self.audio_max_length - ref_audio_ids.shape[1], 0)), value=-100
                    )
                batch["ref_audio_ids"] = ref_audio_ids

        if self.id_column_name is not None:
            batch["id"] = [feature[self.id_column_name] for feature in features]

        return batch



def convert_dataset_str_to_list(
    dataset_names,
    dataset_config_names,
    metadata_dataset_names=None,
    splits=None,
    dataset_samples=None,
    default_split="train",
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")
        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        dataset_samples = dataset_samples.split("+") if dataset_samples is not None else None
        metadata_dataset_names = metadata_dataset_names.split("+") if metadata_dataset_names is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if metadata_dataset_names is not None and len(metadata_dataset_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one metadata dataset is passed for each dataset, got {len(dataset_names)} datasets and {len(metadata_dataset_names)} metadata datasets."
        )

    if dataset_samples is not None:
        if len(dataset_samples) != len(dataset_names):
            raise ValueError(
                f"Ensure one sample is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_samples)} samples."
            )
        dataset_samples = [float(ds_sample) for ds_sample in dataset_samples]
    else:
        dataset_samples = [None] * len(dataset_names)

    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i] if dataset_config_names is not None else None,
                "split": splits[i],
                "metadata_dataset_name": metadata_dataset_names[i] if metadata_dataset_names is not None else None ,
                "samples": dataset_samples[i],
            }
        )
    return dataset_names_dict




def load_multiple_datasets( 
    accelerator: Accelerator,
    loading_method: str = 'disk',
    dataset_names: Union[List, str] = None,
    dataset_config_names: Union[List, str] = None,
    metadata_dataset_names: Optional[str] = None,
    splits: Optional[Union[List, str]] = None,
    label_column_names: Optional[List] = None,
    stopping_strategy: Optional[str] = "first_exhausted",
    dataset_samples: Optional[Union[List, np.array]] = None,
    streaming: Optional[bool] = False,
    seed: Optional[int] = None,
    id_column_name: Optional[str] = None,
    columns_to_keep: Optional[Set[str]] = None,
    prompt_column_name: Optional[str] = None,
    sampling_rate: Optional[int] = None,
    audio_column_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    to_clean: bool = False,  # Remove outlier data
    num_proc = 16,
    **kwargs,
) -> Union[Dataset, IterableDataset]:
    dataset_names_dict = convert_dataset_str_to_list(
        dataset_names, dataset_config_names, metadata_dataset_names, splits, label_column_names, dataset_samples
    )

    if dataset_samples is not None:
        dataset_samples = [ds_dict["samples"] for ds_dict in dataset_names_dict]
        probabilities = np.array(dataset_samples) / np.sum(dataset_samples)
    else:
        probabilities = None

    all_datasets = []
    # iterate over the datasets we want to interleave
    for dataset_dict in tqdm(dataset_names_dict, desc="Combining datasets..."):
        with accelerator.local_main_process_first():
            if loading_method == 'hf':
                dataset = load_dataset(
                    dataset_dict["name"],
                    dataset_dict["config"],
                    split=dataset_dict["split"],
                    streaming=streaming,
                    **kwargs,
                )
            elif loading_method == 'disk':
                if dataset_dict["name"].startswith('s3://'):
                    disable_proxy()
                    storage_options = {"endpoint_url":'', "key":'', "secret": '', "use_ssl": False}
                    dataset = datasets.load_from_disk(dataset_dict["name"], storage_options=storage_options)[dataset_dict["split"]]
                    enable_proxy()
                else:
                    dataset = datasets.load_from_disk(dataset_dict["name"])[dataset_dict["split"]] 
            elif loading_method == 'json':
                download_config=datasets.DownloadConfig(resume_download=True, max_retries=100)
                dataset = load_dataset(
                    'json',
                    data_files=dataset_dict["name"],
                    split=dataset_dict["split"],
                    streaming=streaming,
                    download_config=download_config,
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                    **kwargs,
                    )
            else:
                raise ValueError(f"Unsupported loading method: {loading_method}")

            dataset_features = dataset.features.keys()

            # metadata_dataset_name = dataset_dict["metadata_dataset_name"]
            # if metadata_dataset_name is not None:
            #     logger.info(
            #         f'Merging {dataset_dict["name"]} - {dataset_dict["split"]} with {metadata_dataset_name} - {dataset_dict["split"]}'
            #     )
            #     metadata_dataset = load_dataset(
            #         metadata_dataset_name,
            #         dataset_dict["config"],
            #         split=dataset_dict["split"],
            #         streaming=streaming,
            #         **kwargs,
            #     )

            #     # TODO(YL): I forgot to create unique ids for MLS english.
            #     # To iterate faster, I bypass the original id check and do another one. - Done once because assuming it won't change next time
            #     # if dataset_dict["name"] == "parler-tts/mls_eng_10k":
            #     #     def concat_ids(book_id, speaker_id, begin_time):
            #     #         return {"id": f"{book_id}_{speaker_id}_{str(begin_time).replace('.', '_')}"}
            #     #     dataset = dataset.map(concat_ids, input_columns=["book_id", "speaker_id", "begin_time"], num_proc=24)
            #     #     metadata_dataset = metadata_dataset.map(concat_ids, input_columns=["book_id", "speaker_id", "begin_time"], num_proc=24)
            #     #     metadata_dataset = metadata_dataset.rename_column(id_column_name, f"metadata_{id_column_name}")

            #     if dataset_dict["name"] not in {"parler-tts/mls_eng_10k", "parler-tts/mls_eng"}:
            #         if id_column_name is not None and id_column_name not in dataset.column_names:
            #             raise ValueError(
            #                 f"id_column_name={id_column_name} but has not been found in the dataset columns"
            #                 f"- one of {', '.join(list(dataset.column_names))}."
            #             )
            #         if id_column_name is not None and id_column_name not in metadata_dataset.column_names:
            #             raise ValueError(
            #                 f"id_column_name={id_column_name} but has not been found in the metadata dataset columns"
            #                 f"- one of {', '.join(list(metadata_dataset.column_names))}."
            #             )
            #         elif id_column_name is not None:
            #             metadata_dataset = metadata_dataset.rename_column(id_column_name, f"metadata_{id_column_name}")

            #     metadata_columns_to_remove = set(metadata_dataset.column_names).intersection(set(dataset.column_names))

            #     if prompt_column_name is not None:
            #         # We might have applied some transformations to the prompts (e.g  punctuation restoration)
            #         # so we make sure to remove it from the original dataset
            #         if prompt_column_name in dataset.column_names:
            #             logger.info(
            #                 f"REMOVE {prompt_column_name} from dataset {dataset_dict['name']} - dataset_dict['split']"
            #             )
            #             dataset.remove_columns(prompt_column_name)

            #     metadata_columns_to_remove = set(metadata_dataset.column_names).intersection(set(dataset.column_names))
            #     metadata_dataset = metadata_dataset.remove_columns(metadata_columns_to_remove)

            #     dataset = concatenate_datasets([dataset, metadata_dataset], axis=1)

            #     if id_column_name is not None and dataset_dict["name"] not in {
            #         "parler-tts/mls_eng_10k",
            #         "parler-tts/mls_eng",
            #     }:
            #         if (
            #             len(
            #                 dataset.filter(
            #                     lambda id1, id2: id1 != id2,
            #                     input_columns=[id_column_name, f"metadata_{id_column_name}"],
            #                 )
            #             )
            #             != 0
            #         ):
            #             raise ValueError(
            #                 f"Concatenate didn't work. Some ids don't correspond on dataset {dataset_dict['name']}"
            #             )

            #     dataset_features = dataset.features.keys()

            if columns_to_keep is not None:
                dataset = dataset.remove_columns(set(dataset_features - columns_to_keep))
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        # we have a single dataset so just return it as is
        return all_datasets[0]

    if streaming:
        interleaved_dataset = interleave_datasets(
            all_datasets,
            stopping_strategy=stopping_strategy,
            probabilities=probabilities,
            seed=seed,
        )
    else:
        with accelerator.local_main_process_first():
            interleaved_dataset = concatenate_datasets(all_datasets)

    return interleaved_dataset

