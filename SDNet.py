import torch.nn as nn
import torch
from Unet import UNet
from torch.distributions import Normal
import torch.nn.functional as F

# Architecture from "Disentangled representation learning in cardiac image analysis", Chartsias et al.


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=(3, 3)):
        super(ConvolutionalBlock, self).__init__()
        self.forward_pass = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                      stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1,
                      stride=stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, data):
        return self.forward_pass(data)


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, grad):
        return grad


class FAnatomy(nn.Module):
    def __init__(self, feature_maps, n_a, binary_threshold, device):
        super(FAnatomy, self).__init__()
        self.unet = UNet(num_classes=n_a, num_levels=5,
                         num_filters=[num_filters*feature_maps for num_filters in [16, 32, 64, 128, 256]],
                         device=device)
        self.bin = binary_threshold

    def forward(self, data):
        data = self.unet(data)
        data = nn.Softmax(dim=1)(data)
        if self.bin:
            data = RoundNoGradient.apply(data)
        return data


class FModality(nn.Module):
    def __init__(self, device, n_a, n_z):
        super(FModality, self).__init__()

        self.device = device
        self.conv_blocks = nn.Sequential(
            ConvolutionalBlock(in_channels=n_a + 1, out_channels=16, stride=2).to(self.device),
            ConvolutionalBlock(in_channels=16, out_channels=16, stride=2).to(self.device)
        )

        in_features = 1600
        out_lin = 32
        out_features = n_z

        self.lin1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_lin),
            nn.BatchNorm1d(num_features=out_lin),
            nn.LeakyReLU()
        ).to(self.device)

        # Get mean and log(std) to sample z from
        self.dense_block_mean = nn.Linear(in_features=out_lin, out_features=out_features).to(self.device)
        self.dense_block_log_sigma = nn.Linear(in_features=out_lin, out_features=out_features).to(self.device)

    def forward(self, data):
        data = self.conv_blocks(data).view(data.shape[0], -1)
        data = self.lin1(data)

        # Obtain the mean and log standard deviation
        z_mean = self.dense_block_mean(data)
        z_log_sigma = self.dense_block_log_sigma(data)

        dist = Normal(loc=z_mean, scale=F.softplus(z_log_sigma))

        return dist


class Encoder(nn.Module):
    def __init__(self, n_a, n_z, device, binary_threshold):
        super(Encoder, self).__init__()
        self.device = device
        self.anat = FAnatomy(feature_maps=1, n_a=n_a, binary_threshold=binary_threshold, device=self.device)
        self.mod = FModality(device=self.device, n_a=n_a, n_z=n_z)

    def forward(self, image):
        anatomy = self.anat(image)
        mod_input = torch.cat((image, anatomy), dim=1)
        modality_distribution = self.mod(mod_input)

        return anatomy, modality_distribution


class FiLM(nn.Module):
    def __init__(self, device, n_a, n_z):
        super(FiLM, self).__init__()
        
        self.film = nn.Sequential(
            nn.Linear(in_features=n_z, out_features=n_a*2),
            nn.LeakyReLU(),
            nn.Linear(in_features=n_a*2, out_features=n_a*2)
        ).to(device)
        self.channels = n_a

    def forward(self, residual_input):
        gamma_beta = self.film(residual_input)

        gamma = gamma_beta[:, :self.channels]
        beta = gamma_beta[:, self.channels:]

        return gamma, beta


class FiLMLayer(nn.Module):
    def __init__(self, n_a, n_z, device):
        super(FiLMLayer, self).__init__()

        self.film = FiLM(device, n_a, n_z)
        self.conv_1 = nn.Conv2d(in_channels=n_a, out_channels=n_a, kernel_size=(3, 3), padding=1).to(device)
        self.conv_2 = nn.Conv2d(in_channels=n_a, out_channels=n_a, kernel_size=(3, 3), padding=1).to(device)

    def forward(self, data, modality_factor):
        data = self.conv_1(data)
        data = nn.LeakyReLU(negative_slope=0.3)(data)

        residual = self.conv_2(data)

        gamma, beta = self.film(modality_factor)
        gamma, beta = gamma.view(*gamma.shape, 1, 1), beta.view(*beta.shape, 1, 1)

        residual = residual*gamma + beta
        residual = nn.LeakyReLU(negative_slope=0.3)(residual)

        return data + residual


class Decoder(nn.Module):
    def __init__(self, n_a, n_z, device):
        super(Decoder, self).__init__()
        self.film1 = FiLMLayer(n_a, n_z, device)
        self.film2 = FiLMLayer(n_a, n_z, device)
        self.film3 = FiLMLayer(n_a, n_z, device)
        self.final_block = nn.Conv2d(in_channels=n_a, out_channels=1, kernel_size=(3, 3), padding=1).to(device)

    def forward(self, data, z_vector):
        data = self.film1(data, z_vector)
        data = self.film2(data, z_vector)
        data = self.film3(data, z_vector)
        data = self.final_block(data)
        data = nn.Tanh()(data)
        return data


class SDNet(nn.Module):
    def __init__(self, n_a, n_z, device, binary_threshold=False):
        super(SDNet, self).__init__()
        self.encoder = Encoder(n_a, n_z, device, binary_threshold)
        self.decoder = Decoder(n_a, n_z, device)
        self.device = device
        self.bin = binary_threshold

    def forward(self, image):
        anatomical_factor, modality_distribution = self.encoder(image)
        reconstruction = self.decoder(anatomical_factor, modality_distribution.rsample())

        return anatomical_factor, reconstruction, modality_distribution
