from resnet import resnet18
from BarlowTwins import BarlowTwins, LARS, adjust_learning_rate
from data import custom_data_loader, plot_BarlowTwins_loss
import torch
import os

batch_size = 256
in_channels = 3
encoder = resnet18(in_channels = in_channels)
in_dim = 512
proj_dim = 128
lr_weights = 0.2
lr_biases = 0.0048
epochs = 100
regularizer = 0.0051
weight_decay = 1e-6
epochs = 100

if not os.path.isdir('savedModel'):
	os.mkdir('savedModel')


if torch.cuda.is_available():
	device = torch.device('cuda')
	print("gpu detected for trainig")
else :
	device = torch.device('cpu')
	print("cpu used for training")


model = BarlowTwins(encoder, in_dim = in_dim, proj_dim = proj_dim, regularizer = regularizer)
model = model.to(device)
param_weights = []
param_biases = []
for param in model.parameters():
	if param.ndim == 1:
		param_biases.append(param)
	else :
		param_weights.append(param)
parameters = [{'params' : param_weights}, {'params' : param_biases}]

optimizer = LARS(parameters, lr = 0, weight_decay = weight_decay, 
				weight_decay_filter = True, lars_adaptation_filter = True)

train_loader = custom_data_loader(batch_size, crop_size = 32)


train_loss_list = []
for epoch in range(1, epochs + 1):

	running_average_loss = 0
	epoch_loss = 0

	for step, ((imgs_1, imgs_2), _) in enumerate(train_loader):



		imgs_1 = imgs_1.to(device)
		imgs_2 = imgs_2.to(device)
		adjust_learning_rate(optimizer, train_loader, step, epochs, batch_size, lr_weights, lr_biases)
		optimizer.zero_grad()
		loss = model(imgs_1, imgs_2)
		loss.backward()
		optimizer.step()

		loss = loss.item()
		epoch_loss += loss
		running_average_loss = epoch_loss / (step + 1)
		print("epoch : " + str(epoch) + ", Batch : " + str(step) + " / " + str(len(train_loader)) + ",          bl : " + str(loss) + ",              rl : " + str(running_average_loss))
		train_loss_list.append(epoch_loss)


	plot_BarlowTwins_loss(epoch, train_loss_list)

	# saving the model
	checkpoint_path = "savedModel" + "/model_" + str(epoch) + ".pth"
	torch.save(model.state_dict(), checkpoint_path)







