import torch
from audioldm_eval import EvaluationHelper
import os,glob,shutil,tempfile

def compute_kl_all(real_path,fake_path):
    dataset_list = ['BGM909','SymMV']
    real_audio_files = []
    fake_audio_files = []
    real_temp_dir = tempfile.mkdtemp()
    fake_temp_dir = tempfile.mkdtemp()
    print(f"Real files in: {real_temp_dir}")
    print(f"Fake files in: {fake_temp_dir}")
    with tempfile.TemporaryDirectory() as real_temp_dir, tempfile.TemporaryDirectory() as fake_temp_dir:
        for dataset_name in dataset_list:
            real_dataset_path = f"{real_path}{dataset_name}/wav"
            real_audio_files.extend(glob.glob(os.path.join(real_dataset_path, "*.wav")))
            fake_audio_path = f"{fake_path}/{dataset_name}/wav"
            fake_audio_files.extend(glob.glob(os.path.join(fake_audio_path, "*.wav")))
        for file in real_audio_files:
            shutil.copy(file, os.path.join(real_temp_dir, os.path.basename(file)))
        for file in fake_audio_files:
            shutil.copy(file, os.path.join(fake_temp_dir, os.path.basename(file)))
        score=evaluator.main(fake_temp_dir,real_temp_dir)
    return score

if __name__ == "__main__":
    device = torch.device(f"cuda:{0}")
    evaluator = EvaluationHelper(16000, device, backbone="cnn14")
    #dataset_list=['V2M-Bench']
    dataset_list=['BGM909', 'SymMV']
    model_list=['output_odf']
    real_path='inference/test_dataset/'
    if "V2M-Bench" not in dataset_list:
        print(f"---------------ALL----------------")
        #ALL_metrics = compute_kl_all(real_path, real_path)
        #print(f"GT metrics: {ALL_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_path = f"{model_name}"
            else:
                fake_path = f"compared_models/{model_name}/output"
            ALL_metrics=compute_kl_all(real_path,fake_path)
            print(f"model_name: {model_name}, metrics: {ALL_metrics}")
    for dataset_name in dataset_list:
        print(f"------------{dataset_name}--------------")
        real_audio_path=f"{real_path}{dataset_name}/wav"
        #GT_metrics = evaluator.main(real_audio_path,real_audio_path)
        #print(f"GT metrics: {GT_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_audio_path=f"{model_name}/{dataset_name}/wav"
            else:
                fake_audio_path=f"compared_models/{model_name}/output/{dataset_name}/wav"
            metrics=evaluator.main(fake_audio_path,real_audio_path)
            print(f"model_name: {model_name}, metrics: {metrics}")
