import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num_classes, num_levels, num_filters, device, apply_last_layer=True):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.num_filters = num_filters
        self.apply_last_layer = apply_last_layer
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()

        for i in range(num_levels):
            in_channels = 1 if i == 0 else num_filters[i-1]

            level = nn.Sequential(nn.Conv2d(in_channels, num_filters[i], kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(num_filters[i], num_filters[i], kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True))
            self.encoder_blocks.append(level.to(device))

            if not i == (num_levels - 1):
                self.pooling_layers.append(nn.MaxPool2d(kernel_size=2, stride=2).to(device))

        for i in range(num_levels-2, -1, -1):
            self.upsampling_layers.append(nn.Upsample(scale_factor=2, mode='nearest').to(device))

            in_channels = num_filters[i] + num_filters[i+1]         # Filter concatenated from the encoding path

            level = nn.Sequential(nn.Conv2d(in_channels, num_filters[i], kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(num_filters[i], num_filters[i], kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True))
            self.decoder_blocks.append(level.to(device))

        if self.apply_last_layer:
            self.final_conv = nn.Conv2d(num_filters[0], num_classes, kernel_size=3, stride=1, padding=1).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        conv0 = self.encoder_blocks[0](x)
        out = self.pooling_layers[0](conv0)

        conv1 = self.encoder_blocks[1](out)
        out = self.pooling_layers[1](conv1)

        conv2 = self.encoder_blocks[2](out)
        out = self.pooling_layers[2](conv2)

        conv3 = self.encoder_blocks[3](out)
        out = self.pooling_layers[3](conv3)

        conv4 = self.encoder_blocks[4](out)

        out = self.upsampling_layers[0](conv4)
        upconv3 = self.decoder_blocks[0](torch.cat([out, conv3], dim=1))

        out = self.upsampling_layers[1](upconv3)
        upconv2 = self.decoder_blocks[1](torch.cat([out, conv2], dim=1))

        out = self.upsampling_layers[2](upconv2)
        upconv1 = self.decoder_blocks[2](torch.cat([out, conv1], dim=1))

        out = self.upsampling_layers[3](upconv1)
        upconv0 = self.decoder_blocks[3](torch.cat([out, conv0], dim=1))

        if self.apply_last_layer:
            out = self.final_conv(upconv0)
        else:
            out = upconv0

        return out