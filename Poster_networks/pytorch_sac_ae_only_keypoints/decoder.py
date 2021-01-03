import torch
import torch.nn as nn

from encoder import compute_size


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, output_sizes=None, out_dim=None):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        assert out_dim is not None
        self.out_dim = out_dim

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.output_sizes = output_sizes

        self.deconvs = nn.ModuleList()

        self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))

        for i in range(self.num_layers - 2):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, kernel_size=3, stride=2, padding=1))
        
        self.deconvs.append(nn.ConvTranspose2d(num_filters, obs_shape[0], kernel_size=7, stride=2, padding=3))

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        if self.output_sizes is None:
            for i in range(0, self.num_layers - 1):
                deconv = torch.relu(self.deconvs[i](deconv))
                self.outputs['deconv%s' % (i + 1)] = deconv
            obs = self.deconvs[-1](deconv)
        else:
            for i in range(0, self.num_layers - 1):
                deconv = torch.relu(self.deconvs[i](deconv, output_size=self.output_sizes[self.num_layers - 1 - i]))
                self.outputs['deconv%s' % (i + 1)] = deconv
            obs = self.deconvs[-1](deconv, output_size=self.output_sizes[0])

        
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters, enc
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](        
        obs_shape, feature_dim, num_layers, num_filters, output_sizes=enc.output_sizes, out_dim=enc.out_dim
    )
