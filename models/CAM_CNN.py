import torch
import torch.nn as nn


class CAM_CNN(nn.Module):

    def __init__(self, shape, num_channels):
        super(CAM_CNN, self).__init__()

        self.conv3d = nn.Sequential(nn.Conv3d(in_channels=num_channels, out_channels=32, kernel_size=(3, 3, 3),
                                    stride=1),
                                    nn.ReLU())
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
                                    nn.BatchNorm2d(num_features=32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=1),
                                    nn.ReLU())

        self.gap = nn.AvgPool2d(kernel_size=(4, 4))
        self.num_channels = num_channels
        in_features = self._get_conv_out(shape)
        self.dense = nn.Linear(in_features=in_features, out_features=2)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv3d(torch.zeros(1, self.num_channels, *shape)).squeeze(2)
            o = self.conv2d(o)
            o = self.gap(o).view(1, -1)
            return o.shape[1]

    def forward(self, data):
        data = self.conv3d(data).squeeze(2)
        data = self.conv2d(data)
        data = self.gap(data)
        data = data.view(data.shape[0], -1)  # Vector representation
        data = self.dense(data)
        data = nn.LogSoftmax(1)(data)
        return data
