# -*- coding: utf-8 -*-

import numpy as np
import datetime
from numpy import ndarray
from typing import List, Union, Any, Type, Tuple, NoReturn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import Axes, Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import kornia
import utils
import sys

# Path to folder with script and data files.
Base_folder_path = os.getcwd()

"""## Replay buffer definition"""


# Slightly modified version of: https://github.com/denisyarats/drq/blob/5ad46da6bc185492742f5fbddec4efb829ee1a07/replay_buffer.py#L16
class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self,
                 obs_shape: tuple,
                 capacity: int,
                 image_pad: int = 4,
                 device: str = "cuda"):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self,
            obs: Union[Tensor, ndarray],
            next_obs: Union[Tensor, ndarray]) -> NoReturn:
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.next_obses[self.idx], next_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        idxs2 = np.random.randint(0,
                                  self.capacity if self.full else self.idx,
                                  size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs2]
        # obses_aug = obses.copy()
        # next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        return obses, next_obses


data_cartpole = torch.load(os.path.join(Base_folder_path, "cartpole_swingup_40000_50000.pt"))

DownSampleTo84 = nn.UpsamplingBilinear2d(size=84)
# samples = np.random.randint(0, 10000, 10000)
# data_cartpole[0] = DownSampleTo84(torch.tensor(data_cartpole[0][samples])/255.)
# data_cartpole[1] = DownSampleTo84(torch.tensor(data_cartpole[1][samples])/255.)
data_cartpole[0] = DownSampleTo84(torch.tensor(data_cartpole[0]) / 255.)
data_cartpole[1] = DownSampleTo84(torch.tensor(data_cartpole[1]) / 255.)

torch.manual_seed(0)
permuted_indicies = torch.randperm(data_cartpole[0].shape[0])
input_frames = data_cartpole[0][permuted_indicies, :, :, :]
print(input_frames.shape)
target_frames = data_cartpole[1][permuted_indicies, :, :, :]
print(target_frames.shape)


class input_image_encoder(nn.Module):
    """This is used as an encoder for an image to be used in KeyPoint-learning

    Documenation for each class method can be found in the given method

    Note:
        The default parameters is set as in IMM.py by Nicklas

    Class methods:
        input_layer: Returns a sequential list of one cnn layer
        conv_layers: Returns a sequential list of (n-1) cnn layers

    """

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

        """All arguments are deifned as attributes to be used in the methods.

        Args:
            in_channels (int): The number of channels for the input images
            out_channel (list of int): The output channel for each cnn layer
                                        Note that the len(out_channels) defines
                                        the number of layers
            kernel_size (list of int): The kernel size for each cnn layer
            stride (list of int): The stride for each cnn layer
            padding (list of int): The padding for each cnn layer
            bias (list of boolean): Boolean that defines if bias
                                            should be used in each cnn layer
            batch_norm (list of boolean): Boolean that defines if batch_norm
                                            should be used in each cnn layer
            dilation (list of int): The dilation for each cnn layer
            activations (list of torch activation functions): The activation 
                                        functions for each cnn layer.

        """

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
        """The first cnn layer that encodes the input image

        Args:
            No arguments
        Output:
            A sequential list of:
                                cnn layer
                                Batch normalisation (if True)
                                Activation function

        """

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

        """self.num_layers-1 cnn layers that encodes the input from input_layer

        Args:
            No arguments
        Output:
            A sequential list for each layer of:
                                cnn layer
                                Batch normalisation (if True)
                                Activation function

        """

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

        """self.num_layers-1 cnn layers that encodes the input from input_layer

        Args:
            input_image: Image of size [Batch_norm, Channels, Height, Width]
        Output:
            A list of the original image and the encoded image

        """

        x = self.in_encoder(input_image)
        x = self.out_encoder(x)
        # Returns the original image and the encoded image in a list
        # print(x.device, input_image.device)
        return input_image, x


"""### KeyNet
Network taking a (batch of) image(s) as input and returning the heat maps (and the auxilliary output of the means (i.e. the keypoints) location).
"""


class key_net(nn.Module):
    """This is used to obtain keypoints and corresponding heatmaps from an image

    Documenation for each class method can be found in the given method

    Note:
        The default parameters is set as in IMM.py by Nicklas

    Class methods:
        key_points_from_score_maps: Returns key points (mu) and the probability
                                    distributions used to determine them.
        heat_maps_from_score_maps: Returns heat maps, key points (mu) and the proba-
                                   bility distributions used to determine them.
        forward: Pushes target image through the key_net network and returns
                 the list: [heat_maps, mu, score_maps, probs_x, probs_y]

    """

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


"""### Generator network

Network taking the concatenated feature tensor and heat map tensor as input
"""


class generator_network(nn.Module):
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
                 n_channels_out: int = 9,
                 resolution_in: int = 13,
                 resolution_out: int = 128,
                 device: str = 'cuda'):
        super(generator_network, self).__init__()

        """All arguments are deifned as attributes to be used in the methods.

        Args:
            n_channels_out (int): The number of channels for the resulting images
            n_channels_in_ImEncoder (int): The number of channels in the input tensors from heatmaps
            n_channels_in_Heatmaps (int): The number of channels in the input tensors from encoded images
            resulotion_out (int): The resolution of the resulting images
            device (string): device to compute on

        """
        self.filters_in = n_channels_in_ImEncoder + n_channels_in_Heatmaps
        self.filters_out = n_channels_in_ImEncoder
        self.n_channels_out = n_channels_out
        self.resolution_in = resolution_in
        self.resolution_out = resolution_out
        self.device = device

        self.generator = self.generator_layer()

    def generator_layer(self) -> nn.Sequential:

        """self.num_layers-1 cnn layers that encodes the input from input_layer

        Args:
            No arguments
        Output:
            A sequential list for each layer of:
                                cnn layer
                                Batch normalisation (if True)
                                Activation function
        """

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
    def __init__(self, ImageEncoder_channels: int = 32,
                 n_heatmaps: int = 9,
                 resolution_in: int = 8,
                 resolution_out: int = 84):
        super(KeypointLearning, self).__init__()
        device = 'cuda'
        self.ImageNet = input_image_encoder().to(device)
        self.resolution_in = self.ImageNet(torch.randn((1,9,resolution_out,resolution_out)).to(device))[1].shape[-1]
        self.KeyNet = key_net(num_keypoints=n_heatmaps).to(device)
        self.GenNet = generator_network(n_channels_in_ImEncoder=ImageEncoder_channels,
                                        n_channels_in_Heatmaps=n_heatmaps,
                                        n_channels_out=9,
                                        resolution_in=self.resolution_in,
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


if __name__ == "__main__":
    num_epochs = 50000
    learning_rate = 1e-3  # you know this by now
    val_freq = 10  # validation frequency
    checkpoint_freq = 1000  # frequency in epochs
    batch_size = 50
    batches_per_epoch_train = 100
    batches_per_epoch_val = 20
    train_set_size = 0.7
    val_set_size = 0.2

    loss = nn.MSELoss(reduction='mean')

    device = 'cuda'
    image_pad = 4

    n_sample = input_frames.shape[0]
    train_set_idx = int(train_set_size * n_sample)
    val_set_idx = train_set_idx + int(val_set_size * n_sample)

    rb_cartpole_train = ReplayBuffer(input_frames.shape[1:], input_frames.shape[0],
                                     image_pad=image_pad, device=device)

    rb_cartpole_val = ReplayBuffer(input_frames.shape[1:], input_frames.shape[0],
                                   image_pad=image_pad, device=device)

    rb_cartpole_test = ReplayBuffer(input_frames.shape[1:], input_frames.shape[0],
                                    image_pad=image_pad, device=device)

    for i in range(train_set_idx):
        rb_cartpole_train.add(input_frames[i].numpy(), target_frames[i].numpy())

    for i in range(train_set_idx, val_set_idx):
        rb_cartpole_val.add(input_frames[i].numpy(), target_frames[i].numpy())

    for i in range(val_set_idx, input_frames.shape[0]):
        rb_cartpole_test.add(input_frames[i].numpy(), target_frames[i].numpy())

    """### Training - loop"""

    net = KeypointLearning(n_heatmaps=9).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    cur_epoch = 0

    checkpoint_id = sys.argv[1]

    net.train()
    for epoch in range(cur_epoch, num_epochs + cur_epoch):
        epoch_losses = []

        # Train the network for a single epoch
        for minibatch_i in range(batches_per_epoch_train):

            minibatch_input, minibatch_target = rb_cartpole_train.sample(batch_size)

            output = net(minibatch_input, minibatch_target)
            minibatch_est = output[0]

            minibatch_loss = loss(minibatch_est, minibatch_target)

            if epoch == 0 and minibatch_i == 0:
                print('First loss', minibatch_loss.cpu().detach().numpy())

            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()

            epoch_losses.append(minibatch_loss.cpu().detach().numpy())

        # Evaluate average train loss over the epoch
        train_losses.append(np.mean(epoch_losses))

        # Print current epoch number and train loss
        if (epoch + 1) % 5 == 0:
            print("Epoch: {0:5d}\tTRAIN: {1}".format(epoch + 1, train_losses[-1]))

        # Evaluate validation loss
        if (epoch + 1) % val_freq == 0:
            net.eval()
            val_epoch_losses = []
            for minibatch_i in range(batches_per_epoch_train):
                minibatch_input, minibatch_target = rb_cartpole_val.sample(batch_size)

                output = net(minibatch_input, minibatch_target)
                minibatch_est = output[0]

                minibatch_loss = loss(minibatch_est, minibatch_target)
                val_epoch_losses.append(minibatch_loss.cpu().detach().numpy())

            val_losses.append(np.mean(val_epoch_losses))
            print("Epoch: {0:5d}\tVAL: {1}".format(epoch + 1, val_losses[-1]))
            net.train()

        # Save checkpoint if checkpoint_freq epochs have passed since last checkpoint
        current_time = datetime.datetime.now()
        if (epoch + 1) % checkpoint_freq == 0:
            time_stamp = current_time.strftime("%d_%m_%YT%H_%M_%S")
            checkpoint_name = "keypointnet_ts_{}_epoch_{}_id_{}.pt".format(time_stamp, epoch+1, checkpoint_id)
            checkpoint_path = os.path.join(Base_folder_path, "Checkpoints", checkpoint_name)
            print("Saving checkpoint: {}".format(checkpoint_name))
            torch.save({'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_freq': val_freq
                        }, checkpoint_path)
