import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Encoder_patch(nn.Module):
    def __init__(self, n_channels, emb_size= 512, bilinear=True):
        super(Encoder_patch, self).__init__()
        self.n_channels = n_channels
        self.emb_size   = emb_size
        self.bilinear   = bilinear

        self.conv1 = DoubleConv(n_channels, 128)
        self.conv2 = DoubleConv(128, 256)
        self.conv3 = DoubleConv(256, emb_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   
        x = self.conv3(x) 
        x = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, 1), start_dim = 1)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
#         self.encoder = Encoder_patch(n_channels = in_channels, emb_size= emb_size)
        self.projection = nn.Sequential(
            Rearrange('b c (ph h) (pw w) -> b c (ph pw) h w', c=in_channels, h=patch_size, ph=img_size//patch_size, w=patch_size, pw=img_size//patch_size),
            Rearrange('b c p h w -> (b p) c h w'),
            Encoder_patch(n_channels = in_channels, emb_size= emb_size),
            Rearrange('(b p) d-> b p d', p = (img_size//patch_size)**2),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
        
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
        
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )        
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])        
        
        
class dependencymap(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_regions: int = 256, patch_size: int = 16, img_size: int = 256, output_ch: int=64, cuda=True):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.emb_size = emb_size
        self.output_ch = output_ch
        self.cuda = cuda
        self.outconv   =  nn.Sequential(
            nn.Conv2d(emb_size, output_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(output_ch),
            nn.Sigmoid()
        )
        self.out2  =  nn.Sigmoid()
                
        self.gpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x_gpool = self.gpool(x)
        coeff  = torch.zeros((x.size()[0], self.emb_size, self.img_size, self.img_size))
        coeff2 = torch.zeros((x.size()[0], 1, self.img_size, self.img_size))
        if self.cuda:
            coeff  = coeff.cuda()
            coeff2 = coeff2.cuda()
        for i in range(0, self.img_size//self.patch_size):
            for j in range(0, self.img_size//self.patch_size):
                value = x[:,(i*self.patch_size)+j]
                value = value.view(value.size()[0], value.size()[1], 1, 1)
                coeff[:,:,self.patch_size*i:self.patch_size*(i+1),self.patch_size*j:self.patch_size*(j+1)] = value.repeat(1, 1, self.patch_size, self.patch_size)
                
                value = x_gpool[:,(i*self.patch_size)+j]
                value = value.view(value.size()[0], value.size()[1], 1, 1)
                coeff2[:,:,self.patch_size*i:self.patch_size*(i+1),self.patch_size*j:self.patch_size*(j+1)] = value.repeat(1, 1, self.patch_size, self.patch_size)                
                
        global_contexual      = self.outconv(coeff)
        regional_distribution = self.out2(coeff2)
        return [global_contexual, regional_distribution, self.out2(x_gpool)]
    
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 1024,
                img_size: int = 256,
                depth: int = 2,
                n_regions: int = (256//16)**2,
                output_ch: int = 64, 
                cuda = True, 
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            dependencymap(emb_size, n_regions, patch_size, img_size, output_ch, cuda)
        )     
        
        
        
                
        
        
        
        
        
        
                