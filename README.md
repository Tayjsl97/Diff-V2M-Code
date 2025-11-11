# Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation
This is the official implementation of **Diff-V2M (AAAI'26)**, which is a hierarchical diffusion model with explicit rhythmic modeling and multi-view feature conditioning, achieving state-of-the-art results in video-to-music generation.
- [Paper](https://arxiv.org/abs/2312.10307)
- Check our [demo page](https://tayjsl97.github.io/Diff-V2M-Demo/) and listen!🎧<br>

<img src="img/model.jpg" width="770" height="400" alt="model"/>

## Environment Setup
- Create Anaconda Environment:
  Python 3.9, PyTorch 2.1.0.
  
  ```bash
  git clone https://github.com/Tayjsl97/Diff-V2M.git
  cd Diff-V2M
  ```
  
  ```bash
  conda create -n diff-v2m python=3.9
  conda activate diff-v2m
  pip install -r requirements.txt
  ```
## Pretrained Weights
For training Diff-V2M from scratch, please download the [stable-audio-open-1.0 model](https://huggingface.co/stabilityai/stable-audio-open-1.0/resolve/main/model.safetensors), put them into the directory './saved_model/stable_audio/'.

For inference with Diff-V2M, please download the Diff-V2M model checkpoint [model.ckpt](https://huggingface.co/TaylorJi/Diff-V2M/blob/main/model_odf.ckpt) and its corresponding [model_config.json](https://huggingface.co/TaylorJi/Diff-V2M/blob/main/model_config_odf.json), put them into the directory './saved_model'.
```bash
  mkdir -p saved_model
  wget https://huggingface.co/TaylorJi/Diff-V2M/blob/main/model_config_odf.json -O model_config.json
  wget https://huggingface.co/TaylorJi/Diff-V2M/blob/main/model_odf.ckpt -O model.ckpt
  ```
## Data preperation
Before running the training or inference script, make sure to construct training, validation, and inference datasets. 
After data preprocessing, the json file of dataset looks like:
```json
{
    "dataset_type": "audio_video_dir",
    "rhythm_type": "odf",
    "drop_last": false,
    "datasets": [
        {
            "id": "V2M-Bench",
            "path": "inference/test_dataset/V2M-Bench/wav/",
            "video_feat": "inference/test_dataset/V2M-Bench/clip/",
            "video_info": "inference/test_dataset/V2M-Bench/videoInfo_30.json",
            "video_color": "inference/test_dataset/V2M-Bench/color/",
            "class_label": 2
        }
    ],
    "random_crop": true
}
```
## Training
  Before running the training script, make sure to define the following parameters in `train.sh`:
- `--model-config`
  - Path to the model config file for Diff-V2M
- `--dataset-config`
  - Path to the dataset config file for training
- `--val-dataset-config`
  - Path to the dataset config file for validation
- `--config-file`
  - The path to the defaults.ini file in the repo root, required if running `train.py` from a directory other than the repo root
- `--pretransform-ckpt-path`
  - Used in various model types such as latent diffusion models to load a pre-trained autoencoder. Requires an unwrapped model checkpoint. For training Diff-V2M, this path is 'saved_models/stable_audio/model.safetensors'.
- `--save-dir`
  - The directory in which to save the model checkpoints
- `--checkpoint-every`
  - The number of steps between saved checkpoints.
  - *Default*: 10000
- `--batch-size`
  - Number of samples per-GPU during training. Should be set as large as your GPU VRAM will allow.
  - *Default*: 8
- `--num-gpus`
  - Number of GPUs per-node to use for training
  - *Default*: 1
- `--num-nodes`
  - Number of GPU nodes being used for training
  - *Default*: 1
- `--accum-batches`
  - Enables and sets the number of batches for gradient batch accumulation. Useful for increasing effective batch size when training on smaller GPUs.
- `--strategy`
  - Multi-GPU strategy for distributed training. Setting to `deepspeed` will enable DeepSpeed ZeRO Stage 2.
  - *Default*: `ddp` if `--num_gpus` > 1, else None
- `--precision`
  - floating-point precision to use during training
  - *Default*: 16
- `--num-workers`
  - Number of CPU workers used by the data loader
- `--seed`
  - RNG seed for PyTorch, helps with deterministic training
  
- Start training:
  
  ```bash
  sbatch train.sh
  ```
## Inference
  Before running the inference script, make sure to define the following parameters in `infer.sh`:
- `--model_config_path`
  - Path to the model config file for a local model
- `--dataset_config_path`
  - Path to the dataset config file for inference
- `--ckpt_path`
  - Path to the saved models of Diff-V2M
- `--output_dir`
  - Path to save the generated results

- Run the inference using the following script:

  ```bash
  sbatch infer.sh
  ```
## Reference
If you find the code useful for your research, please consider citing
```bib
@inproceedings{ji2026diff,
  title={Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation},
  author={Ji, Shulei and Wang, Zihao and Yu, Jiaxing and Yang, Xiangyuan and Li, Shuyu and Wu, Songruoyao and Zhang, Kejun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
## Acknowledgments
We appreciate [stable audio open](https://github.com/Stability-AI/stable-audio-tools) for providing the reference codes of audio generation models.

