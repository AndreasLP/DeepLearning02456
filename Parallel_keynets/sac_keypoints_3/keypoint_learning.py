import numpy as np
from numpy import ndarray
from typing import List, Union, Any, Type, Tuple, NoReturn
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import kornia
import utils

from encoder import make_encoder

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class input_image_encoder(nn.Module):
    def __init__(self,
                 in_channels: int = 9,
                 out_channels: List[int] = [32, 32, 32, 32],
                 kernel_size: List[int] = [7, 3, 3, 3],
                 stride: List[int] = [2, 2, 2, 1],
                 padding: List[int] = [3, 1, 1, 1],
                 bias: List[bool] = [True, True, True, True],
                 batch_norm: List[bool] = [True, True, True, True],
                 dilation: List[int] = [1, 1, 1, 1],
                 activations: List[Any] = [nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU]):
        super(input_image_encoder, self).__init__()

        self.num_layers = len(out_channels)  # Number of layers
        self.in_channels = in_channels  # Input channel
        self.out_channels = out_channels  # Output channels
        self.kernel_size = kernel_size  # Kernel size
        self.stride = stride  # Stride
        self.padding = padding  # Padding
        self.bias = bias  # Bias
        self.batch_norm = batch_norm  # Batch norm
        self.dilation = dilation  # Dilation
        self.activations = activations  # Activation functions

        self.in_encoder = self.input_layer()  # The initial encoder
        self.out_encoder = self.conv_layers()  # The self.num_layers-1 cnn layers

    def input_layer(self) -> nn.Sequential:
        input_layer = []  # The sequential list

        # The cnn layer
        input_layer.append(nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=self.out_channels[0],
                                     kernel_size=self.kernel_size[0],
                                     stride=self.stride[0],
                                     padding=self.padding[0],
                                     dilation=self.dilation[0],
                                     bias=self.bias[0]))

        # Batch normalisation
        if self.batch_norm[0]:
            input_layer.append(nn.BatchNorm2d(self.out_channels[0]))

        # Applying the activation function
        if self.activations[0] == nn.ReLU:
            input_layer.append(self.activations[0](inplace=True))  # Use inplace if possible (only for ReLU)
        else:
            input_layer.append(self.activations[0]())  # For all other activation functions

        return nn.Sequential(*input_layer)  # Covert to and return a sequential list

    def conv_layers(self) -> nn.Sequential:

        layers = []  # The sequential list

        # Setting up (self.num_layers-1) cnn layers
        for i in range(1, self.num_layers):
            # The cnn layer
            layers.append(nn.Conv2d(in_channels=self.out_channels[i - 1],
                                    out_channels=self.out_channels[i],
                                    kernel_size=self.kernel_size[i],
                                    stride=self.stride[i],
                                    padding=self.padding[i],
                                    dilation=self.dilation[i],
                                    bias=self.bias[i]))

            # Batch normalisation
            if self.batch_norm[i]:
                layers.append(nn.BatchNorm2d(self.out_channels[i]))

            # Applying the activation function
            if self.activations[i] == nn.ReLU:
                layers.append(self.activations[i](inplace=True))  # Use inplace if possible (only for ReLU)
            else:
                layers.append(self.activations[i]())  # For all other activation functions

        return nn.Sequential(*layers)  # Covert to and return a sequential list

    def forward(self, input_image: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.in_encoder(input_image)
        x = self.out_encoder(x)
        return input_image, x


class key_net(nn.Module):
    def __init__(self,
                 in_channels: int = 9,
                 out_channels: List[int] = [32, 32, 32, 32],
                 kernel_size: List[int] = [7, 3, 3, 3],
                 stride: List[int] = [2, 2, 2, 1],
                 padding: List[int] = [3, 1, 1, 1],
                 bias: List[bool] = [True, True, True, True],
                 batch_norm: List[bool] = [True, True, True, True],
                 dilation: List[int] = [1, 1, 1, 1],
                 activations: List[Any] = [nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU],
                 num_keypoints: int = 10,
                 sigma: float = 0.1):
        super(key_net, self).__init__()

        self.image_encoder = input_image_encoder(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 bias,
                                                 batch_norm,
                                                 dilation,
                                                 activations)
        self.features_to_score_maps = nn.Conv2d(in_channels=out_channels[-1],
                                                out_channels=num_keypoints,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                dilation=1,
                                                bias=True)
        self.num_keypoints = num_keypoints
        self.sigma = sigma

    def key_points_from_score_maps(self, score_maps: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_coor_vector = torch.linspace(-1, 1, score_maps.shape[3], device=score_maps.device)
        y_coor_vector = torch.linspace(-1, 1, score_maps.shape[2], device=score_maps.device)
        probs_x = F.softmax(score_maps.mean(axis=2), dim=2)
        probs_y = F.softmax(score_maps.mean(axis=3), dim=2)
        mu_x = torch.sum(probs_x * x_coor_vector, dim=2)
        mu_y = torch.sum(probs_y * y_coor_vector, dim=2)
        mu = torch.cat((mu_x.unsqueeze(dim=2), mu_y.unsqueeze(dim=2)), dim=2)
        return mu, probs_x, probs_y

    def heat_maps_from_score_maps(self, score_maps: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, probs_x, probs_y = self.key_points_from_score_maps(score_maps)

        # The remaining part of this function is very similar to Nicklas' implementation
        mu_x, mu_y = mu[:, :, 0:1].unsqueeze(dim=3), mu[:, :, 1:2].unsqueeze(dim=3)

        x_coor_vector = torch.linspace(-1, 1, score_maps.shape[3], device=score_maps.device)
        y_coor_vector = torch.linspace(-1, 1, score_maps.shape[2], device=score_maps.device)
        x = x_coor_vector.reshape((1, 1, 1, len(x_coor_vector)))
        y = y_coor_vector.reshape((1, 1, len(y_coor_vector), 1))

        g_x = (x - mu_x) ** 2
        g_y = (y - mu_y) ** 2
        heat_maps = torch.exp((g_x + g_y) * (-1 / (2 * self.sigma ** 2)))

        return heat_maps, mu, probs_x, probs_y

    def forward(self, input_image: Tensor) -> List[Tensor]:
        features = self.image_encoder(input_image)[1]
        score_maps = self.features_to_score_maps(features)
        heat_maps, mu, probs_x, probs_y = self.heat_maps_from_score_maps(score_maps)

        return [heat_maps, mu, score_maps, probs_x, probs_y]

class generator_network(nn.Module):
    def __init__(self,
                 n_channels_in_ImEncoder: int,
                 n_channels_in_Heatmaps: int,
                 n_channels_out: int = 9,
                 resolution_in: int = 13,
                 resolution_out: int = 128,
                 device: str = 'cuda'):
        super(generator_network, self).__init__()

        self.filters_in = n_channels_in_ImEncoder + n_channels_in_Heatmaps
        self.filters_out = n_channels_in_ImEncoder
        self.n_channels_out = n_channels_out
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out
        self.device = device

        self.generator = self.generator_layer()

    def generator_layer(self) -> nn.Sequential:

        layers = []  # The sequential list
        image_size = self.resolution_in
        final_image_size = self.resolution_out
        filters_in = self.filters_in
        filters_out = self.filters_out
        n_channels_out = self.n_channels_out

        # First layer
        layers.append(nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(filters_out))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(filters_out, filters_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(filters_out))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        image_size *= 2
        filters_in = filters_out
        filters_out = int(filters_out / 2)
        # Following layers
        while image_size <= final_image_size:
            layers.append(nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(filters_out))
            layers.append(nn.ReLU(inplace=True))
            if image_size == final_image_size:
                layers.append(nn.Conv2d(filters_out, n_channels_out, kernel_size=3, stride=1, padding=1, bias=True))
                break
            else:
                layers.append(nn.Conv2d(filters_out, filters_out, kernel_size=3, stride=1, padding=1, bias=True))
                layers.append(nn.BatchNorm2d(filters_out))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                image_size *= 2

            if filters_out >= 8:
                filters_in = filters_out
                filters_out = int(filters_out / 2)
            else:
                filters_in = filters_out
        if image_size > final_image_size:
            layers.append(nn.Conv2d(filters_in, n_channels_out, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.UpsamplingBilinear2d(size=final_image_size))
        return nn.Sequential(*layers)

    def forward(self,
                encoded_image: Tensor,
                heatmaps: Tensor) -> Tensor:

        return self.generator(torch.cat((encoded_image, heatmaps), dim=1))


class KeypointLearning(nn.Module):
    def __init__(self, obs_channels: int = 9, 
                 ImageEncoder_channels: int = 128,
                 n_heatmaps: int = 10,
                 resolution_in: int = 8,
                 resolution_out: int = 64):
        super(KeypointLearning, self).__init__()
        device = 'cuda'
        self.ImageNet = input_image_encoder(in_channels=obs_channels).to(device)
        self.KeyNet = key_net(in_channels=obs_channels, num_keypoints=n_heatmaps).to(device)
        self.GenNet = generator_network(n_channels_in_ImEncoder=ImageEncoder_channels,
                                        n_channels_in_Heatmaps=n_heatmaps,
                                        n_channels_out=obs_channels,
                                        resolution_in=resolution_in,
                                        resolution_out=resolution_out).to(device)

    def forward(self,
                input_image: Tensor,
                target_image: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        imagenet_output = self.ImageNet(input_image)
        encoded_image = imagenet_output[1]

        keynet_output = self.KeyNet(target_image)
        heatmaps = keynet_output[0]

        generated_target = self.GenNet(encoded_image, heatmaps)

        return generated_target, imagenet_output, keynet_output


def make_keynet(obs_shape=[9, 64, 64], encoder_feature_dim=50, num_layers=4, num_filters=32, num_keypoints=10, sigma=0.1):
    return KeyNet(obs_shape, encoder_feature_dim, num_layers, num_filters, num_keypoints, sigma)

def make_gennet(obs_shape=[9, 64, 64], num_filters=32, num_keypoints=10, encoded_image_size=8):
    return GenNet(n_channels_in_ImEncoder=num_filters, 
                  n_channels_in_Heatmaps=num_keypoints, 
                  n_channels_out = obs_shape[0],
                  resolution_in=encoded_image_size,
                  resolution_out=obs_shape[-1])


def compute_size(hw, paddings, kernels, strides):
    assert len(paddings) == len(kernels) == len(strides)

    for i in range(len(paddings)):
        hw = int((hw + 2 * paddings[i] - kernels[i])/strides[i] + 1)

    return hw  


def make_imm(obs_shape=[3, 64, 64], encoder_feature_dim=50, num_layers=4, num_filters=32, num_keypoints=10, sigma=0.1):
    
    out_dim = compute_size(obs_shape[-1], 
                           paddings=[3] + (num_layers-2)*[1] + [1], 
                           kernels=[7] + (num_layers-2)*[3] + [3], 
                           strides=[2] + (num_layers-2)*[2] + [1]) 
    
    return KeypointLearning(obs_channels = obs_shape[0],
                            ImageEncoder_channels = 32,
                            n_heatmaps = num_keypoints, 
                            resolution_in = out_dim, 
                            resolution_out = obs_shape[-1])

