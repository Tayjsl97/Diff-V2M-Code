from torch import nn
import torch
from .transformer import ExtractorTransformer


def normalize_tensor(x, eps=1e-6):
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True) + eps
    return (x - mean) / std


def min_max_normalize_tensor(x):
    x_min = x.min(dim=(1, 2), keepdim=True)[0]
    x_max = x.max(dim=(1, 2), keepdim=True)[0]

    return (x - x_min) / (x_max - x_min)

class RhythmExtractor(nn.Module):
    def __init__(self,input_dim,output_dim,dim,depth=12,num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.transformer = ExtractorTransformer(
            dim=dim,
            depth=depth,
            dim_heads=dim // num_heads,
            dim_in=input_dim,
            dim_out=output_dim
        )
        self.class_embedding = nn.Embedding(3, input_dim)
        self.scene_embedding = nn.Embedding(2, input_dim)
        self.pseudo_linear = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, class_label,scene_cut=None,pseudo_beat=None,mask=None):
        x = normalize_tensor(x)
        if isinstance(class_label, list) or isinstance(class_label, tuple):
            class_label = torch.LongTensor(class_label).unsqueeze(1).to(x.device)
        if scene_cut is not None:
            scene_cut = scene_cut.long().to(x.device)
            pseudo_beat = pseudo_beat.unsqueeze(-1).to(x.device)
            scene_embedding = self.scene_embedding(scene_cut)
            pseudo_embedding = self.pseudo_linear(pseudo_beat)
            x=x+scene_embedding+pseudo_embedding
        class_embedding = self.class_embedding(class_label)
        x=torch.cat((class_embedding,x),dim=1)
        output = self.transformer(x)[:,1:,:]
        return output
