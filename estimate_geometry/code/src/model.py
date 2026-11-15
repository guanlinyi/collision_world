"""Small spatial RGB+mask predictor for normalized visible 3D coordinates and faces."""
import torch
from torch import nn
import torch.nn.functional as F
def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
class SurfaceNet(nn.Module):
    def __init__(self,input_channels=4,global_context=False):
        super().__init__();self.a=block(input_channels,24);self.b=block(24,48);self.c=block(48,96);self.d=block(144,48);self.e=block(72,24);self.out=nn.Conv2d(24,9,1)
        self.global_context=global_context
        if global_context:
            self.context_norm=nn.LayerNorm(96);self.context_position=nn.Linear(2,96)
            self.context_attention=nn.MultiheadAttention(96,4,batch_first=True)
            nn.init.zeros_(self.context_attention.out_proj.weight);nn.init.zeros_(self.context_attention.out_proj.bias)
    def forward(self,x,return_features=False):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        if self.global_context:
            yy,xx=torch.meshgrid(torch.linspace(-1,1,c.shape[2],device=c.device,dtype=c.dtype),torch.linspace(-1,1,c.shape[3],device=c.device,dtype=c.dtype),indexing='ij')
            position=self.context_position(torch.stack([xx,yy],-1).reshape(1,-1,2));tokens=c.flatten(2).transpose(1,2);query=self.context_norm(tokens)+position
            attended,_=self.context_attention(query,query,tokens,need_weights=False)
            c=c+attended.transpose(1,2).reshape_as(c)
        d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));o=self.out(e)
        result=(.65*torch.tanh(o[:,:3]),o[:,3:])
        return (*result,e) if return_features else result
