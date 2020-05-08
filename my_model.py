from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class My_model(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet50', new_length=None,
                 dropout=0.8, ):
        super(My_model, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout
        self.softmax = nn.Softmax()
        self.input_size = 224

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.dropout)))

        self.inplanes = 768
        self.branch1 = self._prepare_my_model(base_model)
        self.branch2 = self._prepare_my_model(base_model)
        self.branch3 = self._prepare_my_model(base_model)
        self.compact_cov1 = nn.Conv2d(512, 256, kernel_size=3,
                                      padding=1, bias=False)
        self.compact_cov2 = nn.Conv2d(512, 256, kernel_size=3,
                                      padding=1, bias=False)
        self.compact_cov3 = nn.Conv2d(512, 256, kernel_size=3,
                                      padding=1, bias=False)
        self.compact_res_layer = self._make_layer(Bottleneck, 256, 3)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 1))
        if self.dropout == 0:
            self.dropout1 = None
        else:
            self.dropout1 = nn.Dropout(p=self.dropout)
        self.fc1 = nn.Linear(256 * Bottleneck.expansion, num_class)

    def _prepare_my_model(self, base_model):

        pretrained_model = torchvision.models.resnet34(pretrained=True)

        model = nn.Sequential(*list(pretrained_model.children())[:-2])

        return model

    def forward(self, input):

        # x1 = input[:, :3]
        # x2 = input[:, 3:6]
        # x3 = input[:, 6:]
        x1, x2, x3 = torch.split(input, 3, dim=1)

        x1 = self.branch1(x1)
        x1 = self.compact_cov1(x1)
        x2 = self.branch2(x2)
        x2 = self.compact_cov2(x2)
        x3 = self.branch3(x3)
        x3 = self.compact_cov3(x3)

        # print("x1:", np.shape(x1))
        # print("x2:", np.shape(x2))
        # print("x3:", np.shape(x3))

        x = torch.cat([x1, x2, x3], dim=1)
        x.retain_grad()

        # print("x:", np.shape(x))
        x = self.compact_res_layer(x)
        x = self.pooling1(x)

        x = x.view(x.size(0), -1)
        if self.dropout > 0:
            x = self.dropout1(x)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
