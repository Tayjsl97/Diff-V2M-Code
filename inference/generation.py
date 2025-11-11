import torch
import json
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, truncated_logistic_normal_rescaled, DistributionShift
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from safetensors.torch import load_file
import os
import numpy as np
import torch
import typing as tp
import math
from torchaudio import transforms as T
from torch.nn.functional import interpolate

from moviepy.editor import VideoFileClip, AudioFileClip

def merge_video_audio(video_path, audio_path, output_path):
    video = VideoFileClip(video_path).without_audio()
    audio = AudioFileClip(audio_path)
    if video.duration < audio.duration:
        audio = audio.subclip(0, video.duration)
    else:
        video = video.subclip(0, audio.duration)
    final_video = video.set_audio(audio)
    f_path_video_out = Path(output_path)
    if f_path_video_out.name.startswith("-"):
        tmp_name = f"tmp_{f_path_video_out.name}"
        tmp_path = f_path_video_out.with_name(tmp_name)
    else:
        tmp_path = f_path_video_out

    final_video.write_videofile(str(tmp_path), codec='libx264', audio_codec='aac')

    if tmp_path != f_path_video_out:
        os.rename(tmp_path, f_path_video_out)
    print(f"----Final video saved to {f_path_video_out}")


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_model(model_config_path, ckpt_path, extractor_ckpt_path, device):
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    model = model.to(device).eval().requires_grad_(False)

    wrapper = create_training_wrapper_from_config(model_config, model)

    if ckpt_path is not None:
        ckpt_state = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    else:
        ckpt_state = {}

    if extractor_ckpt_path is not None:
        extractor_state = torch.load(extractor_ckpt_path, map_location='cpu')["state_dict"]
    else:
        extractor_state = {}

    full_state = {}
    if ckpt_path!=extractor_ckpt_path:
        for k, v in ckpt_state.items():
            if "extractor" not in k:
                full_state[k] = v
        for k, v in extractor_state.items():
            if "extractor" in k:
                full_state[k] = v
    else:
        for k, v in ckpt_state.items():
            full_state[k] = v
    wrapper.load_state_dict(full_state, strict=False)
    return wrapper.to(device).eval().requires_grad_(False), model_config

from pathlib import Path

def filter(demo,output_dir):
    demo_list=[]
    filename_list = []
    dataset_name_list = []
    for one_demo in demo:
        path = Path(one_demo["path"])
        dataset_name = path.parts[2]
        file_name = path.stem
        path=f"inference/test_dataset/{dataset_name}/wav/{file_name}.wav"
        if os.path.exists(path):
            output_path = os.path.join(output_dir, dataset_name)
            os.makedirs(output_path, exist_ok=True)
            output_wav_path = os.path.join(output_path, f"{file_name}.wav")
            output_video_path = os.path.join(output_path, f"{file_name}.mp4")
            if not (os.path.exists(output_wav_path) and os.path.exists(output_video_path)):
                demo_list.append(one_demo)
                filename_list.append(file_name)
                dataset_name_list.append(dataset_name)
    return demo_list, filename_list, dataset_name_list


def generate_audio(module, model_config, dataset_config_path, device, output_dir, cfg_scale=3.0, demo_steps=250, batch_size=32):

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    if module.diffusion.pretransform is not None:
        sample_size = sample_size // module.diffusion.pretransform.downsampling_ratio
    model = module.diffusion_ema.model if module.diffusion_ema else module.diffusion.model

    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    test_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=4,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False
    )
    cnt=0
    for batch in test_dl:
        demo_cond=batch[1]
        if model_config['model']['diffusion']['config']['attention_weight_type']=="preFiLM_gate":
            class_list = []
            for one in demo_cond:
                class_list.append(one["class"])
            class_label = torch.tensor(class_list).long().to(device)
        else:
            class_label=None

        demo_cond,filename_list, dataset_name_list=filter(demo_cond,output_dir)
        print(f"filename_list: {filename_list}")
        print(f"dataset_name_list: {dataset_name_list}")
        assert(len(filename_list) == len(dataset_name_list))
        if len(filename_list) == 0:
            continue
        noise = torch.randn([len(filename_list), module.diffusion.io_channels, sample_size], device=device)
        noise = noise.to(next(module.diffusion.parameters()).dtype)

        torch.cuda.empty_cache()
        with torch.no_grad(), torch.cuda.amp.autocast():
            cond = module.diffusion.conditioner(demo_cond, device)
            cond_inputs = module.diffusion.get_conditioning_inputs(cond)
            if module.diffusion_objective == "v":
                fakes = sample(
                    model, noise, demo_steps, 0, class_label, **cond_inputs,
                    cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True
                )
            else:
                fakes = sample_discrete_euler(
                    model, noise, demo_steps, **cond_inputs,
                    cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True
                )
            if module.diffusion.pretransform is not None:
                fakes = module.diffusion.pretransform.decode(fakes)
            del noise
            del cond_inputs
            torch.cuda.empty_cache()
            for i in range(fakes.shape[0]):
                output_path = os.path.join(output_dir, dataset_name_list[i])
                os.makedirs(output_path, exist_ok=True)
                output_wav_path = os.path.join(output_path, f"{filename_list[i]}.wav")
                output_video_path = os.path.join(output_path, f"{filename_list[i]}.mp4")
                if os.path.exists(output_wav_path) and os.path.exists(output_video_path):
                    print(f"{dataset_name_list[i]}/{filename_list[i]} already exists, skipping.")
                    continue
                video_path=f"inference/test_dataset/{dataset_name_list[i]}/mp4/{filename_list[i]}.mp4"
                one_fake = fakes[i]
                fakes_out = one_fake.to(torch.float32).div(torch.max(torch.abs(one_fake))).mul(32767).to(torch.int16).cpu()
                os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
                torchaudio.save(output_wav_path, fakes_out, sample_rate)
                print(f"Audio saved to {output_wav_path}")
                try:
                    merge_video_audio(video_path, output_wav_path, output_video_path)
                    print(f"✅ Merge Done: {output_video_path}")
                except Exception as e:
                    print(f"❌ Audio merge failed for {filename_list[i]}: {e}")
        cnt += 1


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default="stable_audio_tools/configs/model_configs/Diff-V2M/model_config_odf.json")
    parser.add_argument("--dataset_config_path", type=str, default="stable_audio_tools/configs/dataset_configs/dataset_inference.json")
    parser.add_argument("--ckpt_path", type=str, default="saved_models/model.ckpt")
    parser.add_argument("--extractor_ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True, help="save the generated audio")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.extractor_ckpt_path is None:
        extractor_ckpt_path = args.ckpt_path
    else:
        extractor_ckpt_path = args.extractor_ckpt_path

    module, model_config = prepare_model(args.model_config_path, args.ckpt_path, extractor_ckpt_path, device)
    generate_audio(module, model_config, args.dataset_config_path, device, output_dir=args.output_dir)

