import torch
from torchvision import utils
import matplotlib.pyplot as plt
from data import custom_data_loader


def show_batch(images):

    img_batch = images
    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose(1,2,0))
    plt.title('Batch from DataLoader')



from data import custom_data_loader
data_loader = custom_data_loader(10, 24)
for idx, (images, _) in enumerate(data_loader):
	labels = torch.tensor([i for i in range(5)])
	labels = labels.repeat(2)
	images = torch.cat(images)

	if idx == 3 :
		plt.figure()
		show_batch((images))
		plt.axis('off')
		plt.show()
		break




def show_batch(batch):

    img_batch = batch[0]
    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy()[::-1].transpose(1,2,0))
    plt.title('Batch from DataLoader')

