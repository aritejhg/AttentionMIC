# Implement baseline models here.

from math import floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ':', num_param)
            total_param += num_param
    # print('Total Parameters: {}'.format(total_param))
    return total_param
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet_T(nn.Module):
    def __init__(self, block = ResidualBlock, layers = [3,4,6,3]):
        super(ResNet_T, self).__init__()
        
        
        
        self.inplanes = 64
        
        
        self.embed = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            self._make_layer(block, 64, layers[0], stride = 1),
            self._make_layer(block, 128, layers[1], stride = 2),
            self._make_layer(block, 256, layers[2], stride = 2),
            self._make_layer(block, 512, layers[3], stride = 2),
            
            nn.AvgPool2d(7, stride=1),
            
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
        )
        
        self.classify = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    
    def forward(self, X):
        out = self.embed(X)
        out = (out+X).mean(1)
        out = torch.sigmoid(self.classify(out))
        return out
    
# class ResNet(nn.Module):
#     def init(self, block = ResidualBlock, layers = [3,4,6,3], num_classes = 20):
#         super(ResNet, self).init()
        
#         self.inplanes = 64
        

#         # self.conv1 = nn.Sequential(
#         #                 nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
#         #                 nn.BatchNorm2d(64),
#         #                 nn.LeakyReLU())
#         # self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
#         # self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
#         # self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
#         # self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
#         # self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        
        
#         self.ResNet = nn.Sequential(nn.Sequential(
#                         nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
#                         nn.BatchNorm2d(64),
#                         nn.LeakyReLU()),        
#         nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
#         self._make_layer(block, 64, layers[0], stride = 1),
#         self._make_layer(block, 128, layers[1], stride = 2),
#         self._make_layer(block, 256, layers[2], stride = 2),
#         self._make_layer(block, 512, layers[3], stride = 2))
        
# #         self.param_count = count_parameters(self)
# #         print(self.param_count)
        
#         # for m in self.modules():
#         #     if isinstance(m, nn.Linear):
#         #         init.xavier_normal_(m.weight.data)
#         #         if m.bias is not None:
#         #             m.bias.data.zero_()
        
#         # self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512, 128)
        
        
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
            
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes))
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
    
    
#     def forward(self, x):

#         x = self.avgpool(x)
#         x = x.view(x.shape[0], -1)
#         out = torch.sigmoid(self.fc(x))
#         return out
    
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 20)
        )
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        out = torch.sigmoid(self.FC(X))
        return out
   
# I used this     
class FC_T(nn.Module):
    def __init__(self):
        super(FC_T, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
        )
        
        self.classify = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, X):
        out = self.embed(X)
        out = (out+X).mean(1)
        out = torch.sigmoid(self.classify(out))
        return out
        
# Same model as above with max pooling instead of mean pooling
# class FC_T(nn.Module):
    # def __init__(self):
        # super(FC_T, self).__init__()
        # self.embed = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
            # # nn.Linear(512, 512),
            # # nn.Dropout(0.5),
            # # nn.LeakyReLU(),
            # # nn.Linear(512, 512),
            # # nn.Dropout(0.5),
            # # nn.LeakyReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
        # )
        
        # self.classify = nn.Linear(128, 20)
        # self.param_count = count_parameters(self)
        # print(self.param_count)
        
        # for m in self.modules():
            # if isinstance(m, nn.Linear):
                # init.xavier_normal_(m.weight.data)
                # if m.bias is not None:
                    # m.bias.data.zero_()
    # def forward(self, X):
        # out = self.embed(X)
        # out,_ = (out+X).max(1)
        # out = torch.sigmoid(self.classify(out))
        # return out


class BaselineRNN_2(nn.Module):
    def __init__(self):
        super(BaselineRNN_2, self).__init__()
        self.rnn = nn.GRU(128, 64, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True)
        self.FC = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
    def forward(self, X):
        out, _ = self.rnn(X)
        out = torch.sigmoid(self.FC(out[:,-1,:]))
        return out
