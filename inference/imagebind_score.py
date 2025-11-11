import os
import sys
import torch
import argparse
import pandas as pd

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from tqdm import tqdm

def safe_load_single_video(video_path, device):
    try:
        #print(data.load_and_transform_video_data([video_path], device).shape)
        return data.load_and_transform_video_data([video_path], device)
    except Exception as e:
        print(f"[Warning] Failed to load {video_path}: {e}")
        return None  # 或者返回一个 zero tensor


def compute_IB(video_dir,audio_dir,model,device):
    cnt = 0
    avg_score_av = 0.0
    audio_files = sorted(os.listdir(audio_dir))
    video_paths = [os.path.join(video_dir, i.replace(".wav", ".mp4")) for i in audio_files]
    audio_paths = [os.path.join(audio_dir, i) for i in audio_files]
    print(len(video_paths), len(audio_paths))
    batch_size = 50
    for i in tqdm(range(0, len(video_paths), batch_size)):
        batch_video_paths = video_paths[i:i + batch_size]
        batch_audio_paths = audio_paths[i:i + batch_size]
        video_tensor_list = []
        for path in batch_video_paths:
            out = safe_load_single_video(path, device)
            if out is not None:
                video_tensor_list.append(out)  # 因为返回的是字典
        video_input = torch.cat(video_tensor_list, dim=0).to(device)
        inputs = {
            ModalityType.VISION: video_input,
            ModalityType.AUDIO: data.load_and_transform_audio_data(batch_audio_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(
            embeddings[ModalityType.VISION],
            embeddings[ModalityType.AUDIO]
        )
        print(f"similarity: {similarity.shape}")
        avg_score_av += similarity.sum().item()
        cnt += similarity.shape[0]
        # avg_score_av += cos(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]).detach().cpu().numpy()
        # cnt += 1

    # for mp4_path, wav_path in tqdm(zip(video_path, audio_path), total=len(video_path)):
    #     if not (os.path.isfile(wav_path) and os.path.isfile(mp4_path)):
    #         continue
    #     print(f'processing {mp4_path}')
    #     inputs = {
    #         ModalityType.VISION: data.load_and_transform_video_data([mp4_path], device),
    #         ModalityType.AUDIO: data.load_and_transform_audio_data([wav_path], device),
    #     }
    #
    #     with torch.no_grad():
    #         embeddings = model(inputs)
    #
    #     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    #     avg_score_av += cos(embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]).detach().cpu().numpy()
    #     cnt += 1

    # avg_score_av /= cnt
    return avg_score_av,cnt

def compute_IB_all(real_path,audio_dir,model,device):
    dataset_list = ['URMP', 'MUSIC', 'AIST', 'TikTok', 'BGM909', 'SymMV']
    dataset_list = ['BGM909','SymMV']
    total_score=0
    cnt=0
    for dataset_name in dataset_list:
        real_video_path = f"{real_path}/{dataset_name}/mp4"
        audio_path = f"{audio_dir}/{dataset_name}/wav"
        score,num=compute_IB(real_video_path,audio_path,model,device)
        total_score += score
        cnt += num
    return total_score/cnt

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    #dataset_list = ['V2M-Bench']
    dataset_list=['BGM909', 'SymMV']
    model_list=['output_odf']
    real_path = 'inference/test_dataset'
    if "V2M-Bench" not in dataset_list:
        print(f"---------------ALL----------------")
        #ALL_metrics = compute_IB_all(real_path, real_path, model, device)
        #print(f"GT metrics: {ALL_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_path = f"{model_name}"
            else:
                fake_path = f"compared_models/{model_name}/output/"
            ALL_metrics = compute_IB_all(real_path, fake_path, model, device)
            print(f"model_name: {model_name}, metrics: {ALL_metrics}")
    for dataset_name in dataset_list:
        print(f"------------{dataset_name}--------------")
        real_audio_path = f"{real_path}/{dataset_name}/wav"
        real_video_path = f"{real_path}/{dataset_name}/mp4"
        #score,num = compute_IB(real_video_path, real_audio_path, model, device)
        #print(f"GT metrics: {score/num}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_audio_path = f"{model_name}/{dataset_name}/wav"
            else:
                fake_audio_path = f"compared_models/{model_name}/output/{dataset_name}/wav"
            score,num = compute_IB(real_video_path, fake_audio_path, model, device)
            print(f"model_name: {model_name}, metrics: {score/num}")


if __name__ == "__main__":
    main()
