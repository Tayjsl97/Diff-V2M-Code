import numpy as np
from prdc import compute_prdc


import os
import laion_clap

import contextlib
import os
import sys
import glob

@contextlib.contextmanager
def suppress_stdout():
    import logging
    logger = logging.getLogger()
    previous_level = logger.level
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = fnull
            logger.setLevel(logging.ERROR)
            yield
        finally:
            sys.stdout = old_stdout
            logger.setLevel(previous_level)


def get_CLAP_embedding(AUDIO_DIR):
    audio_name = os.listdir(AUDIO_DIR)
    audio_files=[]
    embed_list=[]
    cnt=0
    for name in audio_name:
        if name.endswith('.wav'):
            audio_files.append(os.path.join(AUDIO_DIR,name))
        cnt+=1
        if cnt>=100:
            audio_embed = model.get_audio_embedding_from_filelist(x=audio_files)
            embed_list.append(audio_embed)
            cnt=0
            audio_files=[]
    if cnt>0:
        audio_embed = model.get_audio_embedding_from_filelist(x=audio_files)
        embed_list.append(audio_embed)
    audio_embed = np.concatenate(embed_list, axis=0)
    print(audio_embed.shape)
    return audio_embed

def get_density_coverage(real_audio_path,fake_audio_path):
    metrics = compute_prdc(real_features=get_CLAP_embedding(real_audio_path),
                           fake_features=get_CLAP_embedding(fake_audio_path),
                           nearest_k=5)
    return metrics

def get_all_density_coverage(real_path,fake_path):
    dataset_list = ['BGM909','SymMV']
    real_embed=[]
    fake_embed=[]
    for dataset_name in dataset_list:
        real_dataset_path = f"{real_path}{dataset_name}/wav"
        fake_audio_path = f"{fake_path}/{dataset_name}/wav"
        real_audio_embed = get_CLAP_embedding(real_dataset_path)
        fake_audio_embed = get_CLAP_embedding(fake_audio_path)
        real_embed.append(real_audio_embed)
        fake_embed.append(fake_audio_embed)
    real_audio_embed=np.concatenate(real_embed, axis=0)
    fake_audio_embed=np.concatenate(fake_embed, axis=0)
    metrics = compute_prdc(real_features=real_audio_embed,fake_features=fake_audio_embed,nearest_k=5)
    return metrics



if __name__ == "__main__":
    with suppress_stdout():
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # download the default pretrained checkpoint.
    #dataset_list=['V2M-Bench']
    dataset_list=['BGM909', 'SymMV']
    model_list=['output_odf']
    real_path='inference/test_dataset/'
    if "V2M-Bench" not in dataset_list:
        print(f"---------------ALL----------------")
        #ALL_metrics = get_all_density_coverage(real_path, real_path)
        #print(f"GT metrics: {ALL_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_path = f"{model_name}"
            else:
                fake_path = f"compared_models/{model_name}/output/"
            ALL_metrics=get_all_density_coverage(real_path,fake_path)
            print(f"model_name: {model_name}, metrics: {ALL_metrics}")
    for dataset_name in dataset_list:
        print(f"------------{dataset_name}--------------")
        real_audio_path=f"{real_path}{dataset_name}/wav"
        #GT_metrics=get_density_coverage(real_audio_path,real_audio_path)
        #print(f"GT metrics: {GT_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_audio_path = f"{model_name}/{dataset_name}/wav"
            else:
                fake_audio_path=f"compared_models/{model_name}/output/{dataset_name}/wav"
            metrics=get_density_coverage(real_audio_path,fake_audio_path)
            print(f"model_name: {model_name}, metrics: {metrics}")

