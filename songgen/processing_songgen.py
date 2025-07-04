"""
lyrics/Text/Voice processor class for MusicGen
"""
import os
import torch
import librosa
import soundfile as sf
from transformers import AutoTokenizer
from .lyrics_utils.lyrics_tokenizer import VoiceBpeTokenizer
from transformers.models.encodec.feature_extraction_encodec import EncodecFeatureExtractor
from transformers import Wav2Vec2FeatureExtractor
from demucs import pretrained
from demucs.apply import apply_model
from demucs.audio import convert_audio
from songgen import (
    XCodecModel,
    build_delay_pattern_mask,
)

class SongGenProcessor():
    def __init__(self, ckpt_path, device):
        """
        Initializes the SongGenProcessor 
        """
        self.device = device
        self.text_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, padding_side='right')
        self.lyrics_tokenizer = VoiceBpeTokenizer() 
        mert_path = 'm-a-p/MERT-v1-330M'
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path)
        self.demucs = pretrained.get_model("htdemucs").to(device)
        self.feature_extractor =  EncodecFeatureExtractor(sampling_rate=16000)
        self.audio_encoder = XCodecModel()
        self.audio_encoder_special_token_ids ={
            'mask': 1030,
            "random": {"bos": 1031 , "eos": 1032}, 
            "melody": {"bos": 1033, "eos": 1034},
            "drum": {"bos": 1035, "eos": 1036},
            "vocal": {"bos": 1037, "eos": 1038},
            "acc": {"bos":1039 , "eos":1040},
        }
        
        
    
    def __call__(self, text: str, lyrics: str, refaudio_path=None, refaudio_type=None, ref_voice_path=None, start=0, separate=False, padding=True, return_tensors="pt"):
        
        """
        Processes the input text, lyrics, and audio file, and returns the tensors suitable for model input.
        do not support batching yet

        :param text: text description.
        :param lyrics: Lyrics text. English Only.
        :param ref_voice_path: Optional path to the reference voice.
        :param start: The starting time for the reference voice slice.
        :param separate: Whether to perform audio separation.
        :param return_tensors: Whether to return the tensors as PyTorch tensors.
        :return: A dictionary with the model's inputs, ready for inference.
        """
        # Process lyrics and convert them into token IDs. Must be english now!
        prompt_input_ids = [261] + self.lyrics_tokenizer.encode(lyrics.strip().replace('\n', '.'), lang='en') + [0]
        
        # Tokenize the lyrics and pad to max length
        lyrics_inputs = self.text_tokenizer.pad(
            [{"input_ids": prompt_input_ids}],
            return_tensors=return_tensors,
            padding="longest",
        ).to(self.device) 

        # Tokenize the text descriptions 
        text_inputs = self.text_tokenizer(
            text,
            return_tensors=return_tensors,
            padding="longest",
        ).to(self.device)  

        model_inputs = {
            **text_inputs,
            "prompt_input_ids": lyrics_inputs.input_ids,
            "prompt_attention_mask": lyrics_inputs.attention_mask
        }

        # Process reference voice (if provided)
        if ref_voice_path is not None:
            wav, sr = sf.read(ref_voice_path)
            wav = wav.T 
            wav = librosa.to_mono(wav)  # Convert to mono if stereo
            # Slice the audio according to the start and end times
            lidx = int(start * sr)
            ridx = lidx + int(3 * sr)  # Slice a 3-second segment
            wav = wav[lidx:ridx]

            if separate:
                # Since our model only supports reference voices that contain vocals and does not include accompaniment, it is necessary to perform vocal separation for mixed audio.
                demucs_wav = convert_audio(
                    torch.tensor(wav[None], device=self.device).to(torch.float32), 
                    sr,
                    self.demucs.samplerate,  
                    self.demucs.audio_channels 
                )
                sr = self.demucs.samplerate
                stems = apply_model(self.demucs, demucs_wav.unsqueeze(0))  
                wav = stems[0][-1:].sum(0).mean(0).cpu().numpy()  

            if sr != self.mert_processor.sampling_rate:  
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.mert_processor.sampling_rate)
                sr = self.mert_processor.sampling_rate
        
            mert_inputs = self.mert_processor(
                [wav], sampling_rate=self.mert_processor.sampling_rate, return_tensors="pt", padding="max_length", max_length=3*self.mert_processor.sampling_rate
            )
   
            model_inputs['ref_voice_values'] = mert_inputs['input_values'].to(self.device)
            model_inputs['ref_voice_attention_mask'] = mert_inputs['attention_mask'].to(self.device)
        
        if refaudio_path is not None and refaudio_type is not None:
            refaudio_wav, refaudio_sr = sf.read(refaudio_path)
            refaudio_wav = refaudio_wav.T 
            refaudio_wav = librosa.to_mono(refaudio_wav)
            if refaudio_sr != self.feature_extractor.sampling_rate:  
                refaudio_wav = librosa.resample(refaudio_wav, orig_sr=refaudio_sr, target_sr=self.feature_extractor.sampling_rate)
                refaudio_sr = self.feature_extractor.sampling_rate
            with torch.no_grad():
                self.audio_encoder.model.to(self.device)
                codes = self.audio_encoder.encode(input_values=torch.tensor(refaudio_wav, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0), bandwidth=4)["audio_codes"] #(1, bsz, codebooks, seq_len)
                codes = codes.to(self.device).squeeze(0)  # (1, bsz, codebooks, seq_len) -> (1, codebooks, seq_len)
            num_codebooks = codes.shape[-2]
            #apply codebook-delay
            bos = self.audio_encoder_special_token_ids[refaudio_type]['bos']
            eos = self.audio_encoder_special_token_ids[refaudio_type]['eos']
            ref_bos = (torch.ones((1, num_codebooks, 1)) * bos).to(self.device)
            codes = torch.cat([ref_bos, codes], dim=-1)
            _, delay_pattern_mask = build_delay_pattern_mask(
                codes,
                bos_token_id=bos,
                pad_token_id=eos,
                max_length=codes.shape[-1] + num_codebooks,
                num_codebooks=num_codebooks,
            )
            delay_codes = torch.where(delay_pattern_mask == -1, eos, delay_pattern_mask) #( codebooks, seq_len) 
            model_inputs['ref_audio_ids'] = delay_codes.transpose(0,1).unsqueeze(0)  #(bsz, seq_len, codebooks) 
            #Xcode codes and build delay
        return model_inputs