import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from functools import partial
from sklearn.preprocessing import minmax_scale
import contextlib
import io


class PreActivationBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm_layer, act=nn.ReLU, **kwargs):
        super().__init__()
        self.conv_block = self.get_block(in_size, out_size, norm_layer, act, padding)
        self.adapt = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1) if in_size != out_size else nn.Identity()
        
    def get_block(self, in_size, out_size, norm_layer, act, padding):
        # norm -> relu -> pad? -> conv  ->  norm -> relu -> pad? -> conv
        def block_part(in_size, out_size):
            conv_part = [norm_layer(in_size), act()]
            p = 0
            if padding == 'reflect':
                conv_part += [nn.ReflectionPad2d(1)] # Reflection reduces artifacts
            elif padding == 'replicate':
                conv_part += [nn.ReplicationPad2d(1)]
            else:
                p = "same"
            conv_part += [nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, padding=p)]
            return conv_part        
        conv_block = block_part(in_size, out_size) + block_part(out_size, out_size)
        return nn.Sequential(*conv_block)
    def __repr__(self):
        return ""
    def forward(self, x):
        return self.adapt(x) + self.conv_block(x)

def use_bias(norm_layer):
    if isinstance(norm_layer, partial):
        return norm_layer.func != nn.BatchNorm2d
    else:
        return norm_layer != nn.BatchNorm2d

class Generator(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=64, padding="reflect", num_blocks=9, norm_layer=nn.InstanceNorm2d, act=nn.ReLU, **kwargs):
        super().__init__()
        bias = use_bias(norm_layer)
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=in_size, out_channels=hidden_size, kernel_size=7, bias=bias),
                  norm_layer(hidden_size),
                  act()]
        downsampling = [hidden_size * (2 ** i) for i in range(3)]
        layers += [self.get_sampling_block(in_size=downsampling[i], 
                                           out_size=downsampling[i + 1], 
                                           norm_layer=norm_layer, 
                                           act=act, 
                                           bias=bias,
                                           downsampling=True)
                   for i in range(len(downsampling) - 1)]
        
        layers += [PreActivationBlock(downsampling[-1], downsampling[-1], padding, norm_layer, act) for _ in range(num_blocks)]
        
        upsampling = [downsampling[-1] // (2 ** i) for i in range(3)]
        [hidden_size * 4, hidden_size * 2, hidden_size]
        layers += [self.get_sampling_block(in_size=upsampling[i], 
                                           out_size=upsampling[i + 1], 
                                           norm_layer=norm_layer, 
                                           act=act, 
                                           bias=bias,
                                           downsampling=False)
                   for i in range(len(upsampling) - 1)]
        
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(upsampling[-1], out_size, kernel_size=7, padding=0),
                   nn.Tanh()]
        self.layers = nn.Sequential(*layers)
        
    def get_sampling_block(self, in_size, out_size, norm_layer, act, bias, downsampling=True):
        conv_class = nn.Conv2d if downsampling else partial(nn.ConvTranspose2d, output_padding=1)
        block = [conv_class(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=2, padding=1, bias=bias),
                 norm_layer(out_size),
                 act(),
                 ]
        return nn.Sequential(*block)
    
    def forward(self, x):
        return self.layers(x)
    def __repr__(self):
        return ""    

class Discriminator(nn.Module):
    def __init__(self, in_size, norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU):
        super().__init__()
        hidden_sizes = [in_size, 64, 128, 256, 512]
        layers = [self.get_block(in_size=hidden_sizes[i], 
                        out_size=hidden_sizes[i + 1], 
                        norm_layer=norm_layer if i > 0 else None, 
                        act=act) 
                  for i in range(len(hidden_sizes) - 1)]
        layers += [nn.Conv2d(in_channels=hidden_sizes[-1], out_channels=1, kernel_size=1, stride=1, padding=1)]
        self.layers = nn.Sequential(*layers)

    def get_block(self, in_size, out_size, norm_layer, act):
        block = [nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1)]
        if norm_layer:
            block += [norm_layer(out_size)]
        if act:
            block += [act()]
        return nn.Sequential(*block)

    def forward(self, x):
        return self.layers(x).squeeze(-3)
    def __repr__(self):
        return ""
class CycleGAN(nn.Module):
    def __init__(self, gen_params, discr_params):
        super(CycleGAN, self).__init__()
        self.Ga2b = Generator(in_size=3, out_size=3, **gen_params) # A -> B
        self.Gb2a = Generator(in_size=3, out_size=3, **gen_params) # B -> A
        self.Da   = Discriminator(in_size=3, **discr_params) # A -> {0, 1}
        self.Db   = Discriminator(in_size=3, **discr_params) # B -> {0, 1}
    def __repr__(self):
        return ""
def get_norm_layer(name):
    if name == 'batch_norm':
        return partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif name == 'instance_norm':
        return partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    raise NotImplementedError


import sys

class OutputFilter:
    def __init__(self, patterns):
        self.patterns = patterns
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def write(self, text):
        if any(pattern in text for pattern in self.patterns):
            return
        self.original_stdout.write(text)
        
    def flush(self):
        self.original_stdout.flush()

@st.cache_resource
def load_model():
    # model = None
    gen_params = dict(norm_layer=get_norm_layer('instance_norm'))
    discr_params = dict(act=partial(nn.LeakyReLU, negative_slope=0.2), norm_layer=get_norm_layer('instance_norm'))
    model = CycleGAN(gen_params, discr_params)
    checkpoint = torch.load('model.pth', weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model

model = load_model()

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

mean_a = np.array([0.51839874, 0.49518771, 0.33699873])
std_a = np.array([0.22115407, 0.2041303,  0.18823845])
mean_b = np.array([0.41229027, 0.40928956, 0.39267376])
std_b = np.array([0.22338789, 0.2017264,  0.21975849])

def de_normalize_a(img):
    img = img.cpu().numpy().transpose(1, 2, 0)
    # return img
    return minmax_scale(
            (img.reshape(3, -1) + mean_a[:, None]) * std_a[:, None],
            feature_range=(0., 1.),
            axis=1,
        ).reshape(*img.shape)

def de_normalize_b(img):
    img = img.cpu().numpy().transpose(1, 2, 0)
    # return img
    return minmax_scale(
            (img.reshape(3, -1) + mean_b[:, None]) * std_b[:, None],
            feature_range=(0., 1.),
            axis=1,
        ).reshape(*img.shape)

st.title("Photo <-> Van Gogh Style")

direction = st.radio(
    "Select conversion direction:",
    ('Photo -> Van Gogh', 'Van Gogh -> Photo')
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Original Image', use_container_width=True)
    
    inputs = val_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        if direction == 'Photo to Van Gogh':
            outputs = model.Ga2b(inputs)
            output_img = de_normalize_b(outputs[0])
        else:
            outputs = model.Gb2a(inputs)
            output_img = de_normalize_a(outputs[0])
        
    st.image(output_img, caption='Converted Image', use_container_width=True)