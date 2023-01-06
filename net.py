import torch

from torch import nn
from torch.nn import functional as F
import ipdb


# 输入：2 * 512
class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, stride=(1,1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=(1,1), stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,7), stride=(1,2),padding=(1,3))  # 2*256
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride =(1,2), padding = (0,1)) #2* 128
        self.layer1 = nn.Sequential(Residual(64, 64),
                                    Residual(64, 64))
        self.layer2 = nn.Sequential(Residual(64, 128, use_1x1conv=True, stride=(1,2)), # 2*64
                                    Residual(128, 128))
        self.layer3 = nn.Sequential(Residual(128, 256, use_1x1conv=True, stride=(1,2)), #2*32
                                    Residual(256, 256))
        self.layer4= nn.Sequential(Residual(256, 512, use_1x1conv=True, stride=(1,2)), #2*16
                                    Residual(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512,2)

        
    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0],-1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet()
    net.to(device)
    inp = torch.rand((16,1,2,512)).to(device)
    output = net(inp)
    ipdb.set_trace()
    print('finished')

