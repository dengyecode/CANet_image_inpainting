import torch
from torch import nn
from torch.nn import functional as F
import pdb


class Attn(nn.Module):
    def __init__(self, input_channels=256, output_channels=256, groups=4, ksize=3, stride=1, rate=2, softmax_scale=10.,
                 rates=[1, 2, 4, 8]):
        super().__init__()
        self.groups = groups
        for i in range(groups):
            self.__setattr__('conv{}'.format(str(i).zfill(2)), nn.Sequential(
                nn.Conv2d(input_channels, output_channels // groups, kernel_size=3, dilation=rates[i], padding=rates[i]),
                nn.LeakyReLU(0.2, inplace=True))
            )
        self.attn1 = AtnConv(input_channels=input_channels, output_channels=output_channels//2, ksize=ksize, stride=stride, rate=rate, softmax_scale=softmax_scale)
        self.attn2 = AtnConv2(input_channels=input_channels, output_channels=output_channels//2, ksize=ksize, stride=stride, rate=rate, softmax_scale=softmax_scale)

    def forward(self, x1, x2, mask):
        residual = x1
        attn1 = self.attn1(x1, x2, mask)
        attn2 = self.attn2(x1, x2, mask)
        attn = torch.cat([attn1, attn2], dim=1)
        tmp = []
        for i in range(self.groups):
            tmp.append(self.__getattr__('conv{}'.format(str(i).zfill(2)))(attn))
        y = torch.cat(tmp, dim=1)
        y = y + residual
        return y

class AtnConv2(nn.Module):
    def __init__(self, input_channels=256, output_channels=128, ksize=3, stride=1, rate=2, softmax_scale=10.):
        super(AtnConv2, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.rw = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0)
        self.w = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0)
        self.f = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0)

    def forward(self, x1, x2, mask):
        """ Attention Transfer Network (ATN) is first proposed in
            Learning Pyramid Context-Encoder Networks for High-Quality Image Inpainting. Yanhong Zeng et al. In CVPR 2019.
          inspired by
            Generative Image Inpainting with Contextual Attention, Yu et al. In CVPR 2018.
        Args:
            x1: low-level feature maps with larger resolution.
            x2: high-level feature maps with smaller resolution.
            mask: Input mask, 1 indicates holes.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.
        Returns:
            torch.Tensor, reconstructed feature map.
        """
        # get shapes
        x1 = self.rw(x1)
        x1s = list(x1.size())
        f_x2 = self.f(x2)
        w_x2 = self.w(x2)
        x2s = list(f_x2.size())

        # extract patches from low-level feature maps x1 with stride and rate
        #kernel = 2 * self.rate
        raw_w = extract_patches(x1, kernel=self.ksize, stride=self.stride)
        raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], self.ksize, self.ksize)  # B*HW*C*K*K

        # split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # split high-level feature maps x2 for matching
        f_groups = torch.split(f_x2, 1, dim=0)
        # extract patches from x2 as weights of filter
        w = extract_patches(w_x2, kernel=self.ksize, stride=self.stride)
        w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)  # B*HW*C*K*K

        w_groups = torch.split(w, 1, dim=0)


        # extract patches from masks to mask out hole-patches for matching
        m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
        m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
        m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
        mm = (m > 0.5).float()  # (B, HW, 1, 1)
        mm_groups = torch.split(mm, 1, dim=0)

        y = []
        scale = self.softmax_scale
        padding = 0 if self.ksize == 1 else 1
        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; 
            wi : separated patch tensor along batch dimension of back; 
            raw_wi : separated tensor along batch dimension of back; 
            '''
            # matching based on cosine-similarity
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4])
            if torch.cuda.is_available():
                escape_NaN = escape_NaN.cuda()
            # normalize
            wi_normed = wi / torch.max(torch.sqrt((wi * wi).sum([1, 2, 3], keepdim=True)), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
            yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3])


            # apply softmax to obtain

            yi = yi * mi
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mi
            yi = yi.clamp(min=1e-8)


            # attending
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.stride, padding=1) / 4.
            y.append(yi)
        y = torch.cat(y, dim=0)
        y.contiguous().view(x1s)
        # adjust after filling
        return y


class AtnConv(nn.Module):
    def __init__(self, input_channels=256, output_channels=128, ksize=3, stride=1, rate=2, softmax_scale=10.):
        super(AtnConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.rw = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0)
        self.w = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0)
        self.f = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, padding=0)


    def forward(self, x1, x2, mask=None):
        """ Attention Transfer Network (ATN) is first proposed in
            Learning Pyramid Context-Encoder Networks for High-Quality Image Inpainting. Yanhong Zeng et al. In CVPR 2019.
          inspired by
            Generative Image Inpainting with Contextual Attention, Yu et al. In CVPR 2018.
        Args:
            x1: low-level feature maps with larger resolution.
            x2: high-level feature maps with smaller resolution.
            mask: Input mask, 1 indicates holes.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.
        Returns:
            torch.Tensor, reconstructed feature map.
        """
        # get shapes
        x2 = torch.nn.functional.interpolate(x2, size=[32,32], mode='bilinear', align_corners=True)

        x1 = self.rw(x1)
        x1s = list(x1.size())
        w_x2 = self.f(x2)
        rw_x2 = self.w(x2)
        x2s = list(w_x2.size())

        # extract patches from low-level feature maps x1 with stride and rate

        #kernel = self.ksize
        raw_w = extract_patches(rw_x2, kernel=self.ksize, stride=self.stride)
        raw_w = raw_w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)  # B*HW*C*K*K

        # split tensors by batch dimension; tuple is returned
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # split high-level feature maps x2 for matching
        f_groups = torch.split(x1, 1, dim=0)      # B * C*64*64
        # extract patches from x2 as weights of filter
        w = extract_patches(w_x2, kernel=self.ksize, stride=self.stride)
        w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)  # B*HW*C*K*K

        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is not None:
            mask = F.interpolate(mask, size=x2s[2:4], mode='nearest')
        else:
            mask = torch.zeros([1, 1, x2s[2], x2s[3]])
            if torch.cuda.is_available():
                mask = mask.cuda()
        # extract patches from masks to mask out hole-patches for matching
        m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
        m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
        m = m.mean([2, 3, 4]).unsqueeze(-1).unsqueeze(-1)
        mm = (m > 0.5).float()  # (B, HW, 1, 1)
        mm_groups = torch.split(mm, 1, dim=0)

        y = []
        scale = self.softmax_scale
        padding = 0 if self.ksize == 1 else 1
        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; 
            wi : separated patch tensor along batch dimension of back; 
            raw_wi : separated tensor along batch dimension of back; 
            '''
            # matching based on cosine-similarity
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4])
            if torch.cuda.is_available():
                escape_NaN = escape_NaN.cuda()
            # normalize
            wi_normed = wi / torch.max(torch.sqrt((wi * wi).sum([1, 2, 3], keepdim=True)), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
            yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x1s[2], x1s[3])

            # apply softmax to obtain

            yi = yi * mi
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mi
            yi = yi.clamp(min=1e-8)

            # attending
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=1, padding=1) / 4.
            y.append(yi)
        y = torch.cat(y, dim=0)
        y.contiguous().view(x1s)
        # adjust after filling
        return y


# extract patches
def extract_patches(x, kernel=3, stride=1):
    if kernel != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches


