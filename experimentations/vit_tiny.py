import torch
import torch.nn as nn
import numpy as np


"""
class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
    
    return x

"""
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, n_channels):
        super().__init__()

        # 
        self.proj = nn.Conv2d(
            n_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pe[:, :x.size(1)]
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn = Q @ K.transpose(-2, -1)
        attn = attn / (Q.size(-1) ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT_tiny(nn.Module):
    def __init__(
        self,
        d_model=192,
        # n_classes=37,
        img_size=(224, 224),
        patch_size=(14, 14),
        n_channels=3,
        n_heads=3,
        n_layers=12
    ):
        # patch_size = 14, 16 patches, 
        
        super().__init__()

        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        assert d_model % n_heads == 0

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        max_seq_length = n_patches + 1

        self.patch_embedding = PatchEmbedding(
            d_model,
            patch_size,
            n_channels
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            max_seq_length
        )

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x
    

class vit_tiny_classifier(nn.Module):
    def __init__(self, n_classes, d_model=192):
        super().__init__()
        self.model = ViT_tiny()
        
        # for p in model.parameters():
        #     p.requires_grad = False
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_classes)
        )

    def forward(self, x, freeze = False):
        if freeze:
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)
        cls_token = features[:, 0]
        out = self.mlp(cls_token)
        return out
    
    def freeze_vit(self):
        """
            this func will freeze the ViT_tiny part of the model
        """
        for param in self.model.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    model = ViT_tiny()

    params_count = sum(p.numel() for p in model.parameters())
    print(params_count)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)  # Should be (1, 197, 192)

    model = vit_tiny_classifier(n_classes=37)
    output = model(dummy_input)
    print(output.shape)  # Should be (1, 37)