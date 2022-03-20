import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride = 1, is_last = False):
		super(BasicBlock, self).__init__()

		self.is_last = is_last # is last block in the sequence of blocks with same channels
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes * self.expansion :
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes * self.expansion, kernel_size = 1, stride = stride, bias = False),
				nn.BatchNorm2d(planes * self.expansion)
				)

	def forward(self, x):

		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)

		preact = out
		out = F.relu(out)
		if self.is_last :
			return out, preact
		else :
			return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride = 1, is_last = False):
		super(Bottleneck, self).__init__()
		self.is_last = is_last

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1, bias = False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes :
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, planes * self.expansion, kernel_size = 1, stride = stride, bias = False),
				nn.BatchNorm2d(planes * self.expansion)
				)

	def forward(self, x):



		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)

		preact = out
		out = F.relu(out)
		if self.is_last:
			return out, preact
		else :
			return out


class ResNet(nn.Module):

	def __init__(self, block, num_blocks, in_channels):
		super(ResNet, self).__init__()

		self.in_channels = in_channels
		self.in_planes = 64
		self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.bn1 = nn.BatchNorm2d(64)

		self.layer1 = self.make_layer(block, 64, num_blocks[0], stride = 1)
		self.layer2 = self.make_layer(block, 128, num_blocks[1], stride = 2)
		self.layer3 = self.make_layer(block, 256, num_blocks[2], stride = 2)
		self.layer4 = self.make_layer(block, 512, num_blocks[3], stride = 2)
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal(m.weight, mode = 'fan_out', nonlinearity  = 'relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)



	def make_layer(self, block, planes, num_blocks, stride):

		strides = [stride] + [1] * (num_blocks - 1)
		layers = []

		for i in range(num_blocks):
			stride = strides[i]
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):

		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avgpool(out)
		out = torch.flatten(out, 1)
		return out


def resnet18(**kwargs):
	return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def resnet34(**kwargs):
	return ResNet(Bottleneck, [3,4,6,3], **kwargs)


