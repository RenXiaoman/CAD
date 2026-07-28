import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse


class ConvDWT(nn.Module):  # DWT: (B,C,H,W) -> (B,4C,H/2,W/2), no parameters are learnable
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        # one-level DWT
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        # input size: x (B, C, H, W)
        if x.dtype != torch.float32:
            x = x.float()
        Yl, Yh = self.dwt_forward(x)
        b, c, h, w = x.shape
        print("Input shape:", x.shape)
        # Yl (B, C, H/2, W/2) for low-frequency LL
        # List Yh for high-frequency from each level of DWT
        # Yh[0] (B, C, 3, H/2, W/2) for high-frequency LH,HL,HH
        print("Before Yh[0] shape:", Yh[0].shape)
        Yh = Yh[0].transpose(1, 2).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])
        print("After Yh[0] shape:", Yh.shape)
        # output size: output (B, 4C, H/2, W/2)
        output = torch.cat((Yl, Yh), dim=1)
        print("Output shape:", output.shape)
        output = F.interpolate(output, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        print("Output shape:", output.shape)
        return output
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dwt = ConvDWT(wave='haar', mode='symmetric').to(device)
    
    x = torch.randn(5, 16, 128, 128).to(device)
    y = dwt(x)
    print(y.shape)