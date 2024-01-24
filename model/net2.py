from torch import nn
import torch
import math
from model.attn2 import Attn
from torch.nn import functional as F
from model.base_function import init_net



def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=64)
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)



class Generator(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.down0 = RefineBlcok0(in_ch=4, out_ch=ngf, kernel_size=5, stride=1, padding=2)
        self.down1 = RefineBlcok(in_ch=ngf, out_ch=ngf*2, kernel_size=3, stride=2, padding=1)
        self.down11 = RefineBlcok(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.down2 = RefineBlcok(in_ch=ngf*2, out_ch=ngf*4, kernel_size=3, stride=2, padding=1)

        self.middle1 = RefineBlcok2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1)
        self.middle2 = RefineBlcok2(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=2, padding=2)
        self.middle3 = RefineBlcok2(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=3, padding=3)
        self.middle4 = RefineBlcok2(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=2, padding=2)
        self.middle5 = RefineBlcok2(in_ch=ngf*4, out_ch=ngf * 4, kernel_size=3, stride=1, dilation=3, padding=3)
        self.middle6 = RefineBlcok2(in_ch=ngf*4, out_ch=ngf*4, kernel_size=3, stride=1, padding=1)

        self.up1 = RefineBlcok(in_ch=ngf*4, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.up11 = RefineBlcok(in_ch=ngf*2, out_ch=ngf*2, kernel_size=3, stride=1, padding=1)
        self.up2 = RefineBlcok(in_ch=ngf*2, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.up21 = RefineBlcok(in_ch=ngf, out_ch=ngf, kernel_size=3, stride=1, padding=1)
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )
        self.attn = Attn(input_channels=ngf*4, output_channels=ngf*4)



    def forward(self, img_m, mask):
        noise = torch.normal(mean=torch.zeros_like(img_m), std=torch.ones_like(img_m) * (1. / 256.))
        feature = img_m + noise
        feature = torch.cat([feature, mask], dim=1)
        m64 = F.interpolate(mask, size=[64, 64], mode='nearest')
        x = self.down0(feature)
        x = self.down1(x)

        x = self.down11(x)
        x = self.down2(x)

        x = self.middle1(x)
        x = self.middle2(x)
        x = self.middle3(x)
        x = self.middle4(x)
        x = self.middle5(x)
        x = self.attn(x, x,m64)
        x = self.middle6(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up1(x)
        x = self.up11(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up2(x)
        x = self.up21(x)
        x = self.out(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()


        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.sigmoid(x)

        return x, [x]


class RefineBlcok0(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch == in_ch and stride == 1:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
        self.act0 = nn.LeakyReLU(0.2, True)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)

        self.kernel = kernel_size
        self.pad = padding
        self.dilation = dilation

        self.stride = stride
        self.query = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.mask_conv =nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):

        residual = self.projection(x)
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act1(out)
        out = self.refine(out)
        out = residual + out

        return out

    def refine(self, x):

        out1 = self.conv2(x)
        mask = self.mask_conv(x)
        B, C, H, W = x.size()
        x = x * mask
        q = self.query(x)
        pad_x = F.pad(x, [self.pad, self.pad, self.pad, self.pad])
        k = self.key(pad_x)
        v = self.value(pad_x)
        k = k.unfold(2, self.kernel, 1, ).unfold(3, self.kernel, 1).contiguous().view(B, C, H, W, self.kernel*self.kernel)
        sim = (q.unsqueeze(4) * k).sum(dim=1).view(B, H, W, -1) # B,H,W,KH*KW
        attn = F.softmax(sim, dim=3)  # B,H,W,KH*KW
        attn = attn.contiguous().view(B, 1, H, W, -1)
        v = v.unfold(2, self.kernel, 1).unfold(3, self.kernel, 1).contiguous().view(B, C, H, W, -1)
        out2 = torch.einsum('bchwk -> bchw', attn * v).view(B, C, H, W)
        out = out1 * mask + (1 - mask) * out2
        return out


class RefineBlcok(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch == in_ch and stride == 1:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding)
        self.act0 = nn.LeakyReLU(0.2, True)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.n0 = nn.InstanceNorm2d(in_ch, track_running_stats=False)

        self.kernel = kernel_size
        self.pad = padding
        self.dilation = dilation

        self.stride = stride
        self.query = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)

        self.mask_conv =nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):

        residual = self.projection(x)
        out = self.n0(x)
        out = self.act0(out)
        out = self.conv1(out)
        out = self.n1(out)
        out = self.act1(out)
        out = self.refine(out)
        out = residual + out

        return out

    def refine(self, x):

        out1 = self.conv2(x)
        mask = self.mask_conv(x)
        B, C, H, W = x.size()
        x = x * mask
        q = self.query(x)
        pad_x = F.pad(x, [self.pad, self.pad, self.pad, self.pad])
        k = self.key(pad_x)
        v = self.value(pad_x)
        k = k.unfold(2, self.kernel, 1, ).unfold(3, self.kernel, 1).contiguous().view(B, C, H, W, self.kernel*self.kernel)
        sim = (q.unsqueeze(4) * k).sum(dim=1).view(B, H, W, -1) # B,H,W,KH*KW
        attn = F.softmax(sim, dim=3)  # B,H,W,KH*KW
        attn = attn.contiguous().view(B, 1, H, W, -1)
        v = v.unfold(2, self.kernel, 1).unfold(3, self.kernel, 1).contiguous().view(B, C, H, W, -1)
        out2 = torch.einsum('bchwk -> bchw', attn * v).view(B, C, H, W)
        out = out1 * mask + (1 - mask) * out2
        return out


class RefineBlcok2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()

        if out_ch == in_ch and stride == 1:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=1)
        self.act0 = nn.LeakyReLU(0.2, True)
        self.n1 = nn.InstanceNorm2d(out_ch, track_running_stats=False)
        self.n0 = nn.InstanceNorm2d(in_ch, track_running_stats=False)

        self.kernel = kernel_size
        self.pad = 1

        self.stride = stride
        self.query = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.mask_conv =nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        residual = self.projection(x)
        out = self.n0(x)
        out = self.act0(out)
        out = self.conv1(out)
        out = self.n1(out)
        out = self.act1(out)
        out = self.refine(out)
        out = residual + out

        return out

    def refine(self, x):
        out1 = self.conv2(x)
        mask = self.mask_conv(x)
        B, C, H, W = x.size()
        x = x * mask
        q = self.query(x)
        pad_x = F.pad(x, [self.pad, self.pad, self.pad, self.pad])
        k = self.key(pad_x)
        v = self.value(pad_x)
        k = k.unfold(2, self.kernel, 1, ).unfold(3, self.kernel, 1).contiguous().view(B, C, H, W, self.kernel*self.kernel)
        sim = (q.unsqueeze(4) * k).sum(dim=1).view(B, H, W, -1) # B,H,W,KH*KW
        attn = F.softmax(sim, dim=3)  # B,H,W,KH*KW
        attn = attn.contiguous().view(B, 1, H, W, -1)
        v = v.unfold(2, self.kernel, 1).unfold(3, self.kernel, 1).contiguous().view(B, C, H, W, -1)
        out2 = torch.einsum('bchwk -> bchw', attn * v).view(B, C, H, W)
        out = out1 * mask + (1 - mask) * out2
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module