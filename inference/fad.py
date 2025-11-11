from frechet_audio_distance import FrechetAudioDistance
import glob,os,shutil
import tempfile

def compute_FAD_ALL(real_path,fake_path):
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
        score=frechet.score(real_temp_dir,fake_temp_dir,dtype="float32")
    return score

if __name__ == "__main__":
    # to use `vggish`
    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False
    )
    # to use `PANN`
    # frechet = FrechetAudioDistance(
    #     model_name="pann",
    #     sample_rate=16000,
    #     use_pca=False,
    #     use_activation=False,
    #     verbose=False
    # )
    # # to use `CLAP`
    # frechet = FrechetAudioDistance(
    #     model_name="clap",
    #     sample_rate=48000,
    #     submodel_name="630k-audioset",  # for CLAP only
    #     verbose=False,
    #     enable_fusion=False,            # for CLAP only
    # )
    # # to use `EnCodec`
    # frechet = FrechetAudioDistance(
    #     model_name="encodec",
    #     sample_rate=48000,
    #     channels=2,
    #     verbose=False,
    # )
    #dataset_list=['V2M-Bench']
    dataset_list=['BGM909', 'SymMV']
    model_list=['output_odf']
    real_path='inference/test_dataset/'
    if "V2M-Bench" not in dataset_list:
        print(f"---------------ALL----------------")
        #ALL_metrics = compute_FAD_ALL(real_path, real_path)
        #print(f"GT metrics: {ALL_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_path = f"{model_name}"
            else:
                fake_path = f"compared_models/{model_name}/output"
            ALL_metrics=compute_FAD_ALL(real_path,fake_path)
            print(f"model_name: {model_name}, metrics: {ALL_metrics}")
    for dataset_name in dataset_list:
        print(f"------------{dataset_name}--------------")
        real_audio_path=f"{real_path}{dataset_name}/wav"
        #GT_metrics=frechet.score(real_audio_path,real_audio_path,dtype="float32")
        #print(f"GT metrics: {GT_metrics}")
        for model_name in model_list:
            if model_name.startswith('output'):
                fake_audio_path=f"{model_name}/{dataset_name}/wav"
            else:
                fake_audio_path=f"compared_models/{model_name}/output/{dataset_name}/wav"
            metrics=frechet.score(real_audio_path,fake_audio_path,dtype="float32")
            print(f"model_name: {model_name}, metrics: {metrics}")
