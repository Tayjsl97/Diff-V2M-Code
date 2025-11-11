# Diff-V2M
This is the official implementation of **Diff-V2M (AAAI'26)**, which is a hierarchical diffusion model with explicit rhythmic modeling and multi-view feature conditioning, achieving state-of-the-art results in video-to-music generation.
- [Paper](https://arxiv.org/abs/2312.10307)
- Check our [demo page](https://tayjsl97.github.io/Diff-V2M-Demo/) and listen!🎧<br>

<img src="img/model.jpg" width="770" height="400" alt="model"/>

## Environment Setup
- Create Anaconda Environment:
  Python 3.9, PyTorch 2.1.0.
  
  ```bash
  git clone https://github.com/Tayjsl97/Diff-V2M-Code.git; cd Diff-V2M-Code
  ```
  
  ```bash
  conda create -n diff-v2m python=3.9
  conda activate diff-v2m
  pip install -r requirements.txt
  ```
## Pretrained Weights
Please download the Diff-V2M model checkpoint state_dict.ckpt, put them into the directory './saved_model'.
## Training
## Inference

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

