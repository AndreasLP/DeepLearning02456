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


class KeyNet(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, encoder_feature_dim, num_layers=4, num_filters=32, num_keypoints=10, sigma=0.1):
        super().__init__()

        assert len(obs_shape) == 3

        self.num_layers = num_layers

        # keynet's image encoder. Similar to AE's image encoder
        self.image_encoder = make_encoder('pixel', obs_shape, encoder_feature_dim, num_layers, num_filters)
        
        self.features_to_score_maps = nn.Conv2d(in_channels = num_filters, 
                                                out_channels = num_keypoints, 
                                                kernel_size = 1,
                                                stride = 1,
                                                padding = 0,
                                                dilation = 1,
                                                groups=1,
                                                bias = True)
        self.num_keypoints = num_keypoints
        self.sigma = sigma

        self.outputs = dict()

        


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
        mu_x, mu_y = mu[:,:,0:1].unsqueeze(dim=3), mu[:,:,1:2].unsqueeze(dim=3)

        x_coor_vector = torch.linspace(-1, 1, score_maps.shape[3], device=score_maps.device)
        y_coor_vector = torch.linspace(-1, 1, score_maps.shape[2], device=score_maps.device)
        x = x_coor_vector.reshape((1, 1, 1, len(x_coor_vector)))
        y = y_coor_vector.reshape((1, 1, len(y_coor_vector), 1))

        g_x = (x - mu_x) ** 2
        g_y = (y - mu_y) ** 2
        heat_maps = torch.exp((g_x + g_y) * (-1 / (2 * self.sigma ** 2)))

        return heat_maps, mu, probs_x, probs_y


    def forward(self, input_image: Tensor, detach=False) -> List[Tensor]:
        _, features = self.image_encoder.forward_conv(input_image)
        if detach:
            features.detach()
        score_maps = self.features_to_score_maps(features)
        heat_maps, mu, probs_x, probs_y = self.heat_maps_from_score_maps(score_maps)

        return [heat_maps, mu, score_maps, probs_x, probs_y]




class GenNet(nn.Module):
    
    """This is used as a decoder for an image to be used in KeyPoint-learning

    Documenation for each class method can be found in the given method

    Note:
        The default parameters is set as in IMM.py by Nicklas

    Class methods:
        generator_layer: Returns a sequential list of n cnn layers

    """
    
    def __init__(self,
                 n_channels_in_ImEncoder: int,
                 n_channels_in_Heatmaps: int, 
                 n_channels_out: int = 3,
                 resolution_in: int = 13,
                 resolution_out: int = 128, 
                 device: str = 'cuda'):
        super(GenNet, self).__init__()

        self.filters_in = n_channels_in_ImEncoder + n_channels_in_Heatmaps
        self.filters_out = n_channels_in_ImEncoder
        self.n_channels_out = n_channels_out
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out
        self.device = device

        self.generator = self.generator_layer()
        
    def generator_layer(self) -> nn.Sequential:
        
        layers = [] #The sequential list
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
        filters_out = int(filters_out/2)
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


class IMM(nn.Module):
    def __init__(self, obs_shape, encoder_feature_dim, num_layers, num_filters, num_keypoints, sigma):
        super(IMM, self).__init__()
        
        self.ImageEncoder = make_encoder('pixel', obs_shape, encoder_feature_dim, num_layers, num_filters)
        self.KeyNet = make_keynet(obs_shape, encoder_feature_dim, num_layers, num_filters, num_keypoints, sigma)
        self.GenNet = make_gennet(obs_shape, num_filters, num_keypoints, encoded_image_size=self.KeyNet.image_encoder.out_dim)
        self.output = None

    def forward(self, input_image: Tensor,
            target_image: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        _, imagenet_output = self.ImageEncoder.forward_conv(input_image) 
        encoded_image = imagenet_output

        keynet_output = self.KeyNet(target_image)
        heatmaps = keynet_output[0]

        generated_target = self.GenNet(encoded_image, heatmaps)
        self.output = (generated_target, imagenet_output, keynet_output)
        return generated_target, imagenet_output, keynet_output


def make_keynet(obs_shape=[9, 64, 64], encoder_feature_dim=50, num_layers=4, num_filters=32, num_keypoints=10, sigma=0.1):
    return KeyNet(obs_shape, encoder_feature_dim, num_layers, num_filters, num_keypoints, sigma)

def make_gennet(obs_shape=[9, 64, 64], num_filters=32, num_keypoints=10, encoded_image_size=8):
    return GenNet(n_channels_in_ImEncoder=num_filters, 
                  n_channels_in_Heatmaps=num_keypoints, 
                  n_channels_out = obs_shape[0],
                  resolution_in=encoded_image_size,
                  resolution_out=obs_shape[-1])
                             
def make_imm(obs_shape=[9, 64, 64], encoder_feature_dim=50, num_layers=4, num_filters=32, num_keypoints=10, sigma=0.1):
    return IMM(obs_shape, encoder_feature_dim, num_layers, num_filters, num_keypoints, sigma)

