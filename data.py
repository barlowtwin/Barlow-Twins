# transforms, dataloader for CIFAR10 dataset

import random
import os
from PIL import ImageOps, Image, ImageFilter
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets



class ImageTransform :

	def __init__(self, transform):
		self.transform = transform

	def __call__(self, x):
		return [self.transform(x), self.transform(x)]


class GaussianBlur(object):

	def __init__(self, p):
		self.p = p

	def __call__(self, img):

		if random.random() < self.p :
			sigma = random.random() * 1.9 * 0.1
			return img.filter(ImageFilter.GaussianBlur(sigma))

		else :
			return img


class Solarization(object):

	def __init__(self, p):
		self.p = p

	def __call__(self, img):

		if random.random() < self.p :
			return ImageOps.solarize(img)
		else :
			return img


def get_transform(crop_size):

	# mean and std for CIFAR10
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2023, 0.1994, 0.2010)

	color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

	data_transforms = transforms.Compose([
		transforms.RandomResizedCrop(crop_size, interpolation = Image.BICUBIC),
		transforms.RandomHorizontalFlip(p = 0.5),
		transforms.RandomApply([color_jitter], p = 0.8),
		transforms.RandomGrayscale(p = 0.2),
		GaussianBlur(p = 0.1),
		Solarization(p = 0.2),
		transforms.ToTensor(),
		transforms.Normalize(mean = mean, std = std)
		])

	return data_transforms


def custom_data_loader(batch_size, crop_size):

	if not os.path.isdir('data'):
		os.mkdir('data')

	data_transform = get_transform(crop_size = crop_size)
	train_dataset = datasets.CIFAR10('data', transform  = ImageTransform(data_transform), download = True)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, drop_last  = True)
	return train_loader



def plot_BarlowTwins_loss(epochs, losses):

	if not os.path.isdir('Plots'):
		os.mkdir('Plots')
	plt.plot(range(epochs), losses)
	plt.xlabel('Epochs')
	plt.ylabel('Barlow Twins Loss')
	plt.savefig('Plots/BarlowTwinsLoss.jpeg')