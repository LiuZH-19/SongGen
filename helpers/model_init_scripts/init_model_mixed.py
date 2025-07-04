
import argparse
import os
from transformers import AutoConfig
from songgen import (
    SongGenDecoderConfig, 
    SongGenForCausalLM,
    SongGenMixedForConditionalGeneration,
    XCodecModel
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_directory", type=str, help="Directory where to save the model and the decoder.")
    parser.add_argument("--text_model", type=str, help="Repository id or path to the text encoder.")
    parser.add_argument("--audio_model", type=str, default="xcodec", help="Repository id or path to the audio encoder")
    parser.add_argument("--track_pattern", type=str, help="The token pattern used. Mixed Mode: ['mixed', 'mixed_pro']; Dual-track Mode: ['parallel_std', 'parallel_A_V', 'parallel_V_A', 'interleaving_A_V', 'interleaving_V_A']")

    args = parser.parse_args()

    text_model = args.text_model
    encodec_version = args.audio_model

    t5 = AutoConfig.from_pretrained(text_model)
    if encodec_version == 'xcodec':
        encodec = XCodecModel()
    else:
        encodec = AutoConfig.from_pretrained(encodec_version)
    

    encodec_vocab_size = encodec.codebook_size
    num_codebooks = encodec.num_codebooks
    print("num_codebooks", num_codebooks)

    decoder_config = SongGenDecoderConfig(
        vocab_size=encodec_vocab_size + 64,  # + 64 instead of +1 to have a multiple of 64
        max_position_embeddings=6547,  # 30 s = 1500 
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=encodec_vocab_size,
        eos_token_id=encodec_vocab_size,
        bos_token_id=encodec_vocab_size + 1,
        num_codebooks=num_codebooks,
        track_pattern=args.track_pattern,
    )

    decoder = SongGenForCausalLM(decoder_config)
    decoder.save_pretrained(os.path.join(args.save_directory, "decoder"))

    model = SongGenMixedForConditionalGeneration.from_sub_models_pretrained(
        text_encoder_pretrained_model_name_or_path=text_model,
        audio_encoder_pretrained_model_name_or_path=encodec_version,
        decoder_pretrained_model_name_or_path=os.path.join(args.save_directory, "decoder"),
        vocab_size=7000, # voicebpe_tokenizer vocab_size=6681 
        prompt_cross_attention=True,
        add_prenet=True
    )

    # set the appropriate bos/pad token ids
    model.generation_config.decoder_start_token_id = encodec_vocab_size + 1
    model.generation_config.pad_token_id = encodec_vocab_size
    model.generation_config.eos_token_id = encodec_vocab_size

    # set other default generation config params
    model.generation_config.max_length = int(30 * model.audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True  # True

    model.config.pad_token_id = encodec_vocab_size
    model.config.decoder_start_token_id = encodec_vocab_size + 1

    model.save_pretrained(os.path.join(args.save_directory, f"untrained-{args.track_pattern}/"))





