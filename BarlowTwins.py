import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

def off_diagonal(matrix): # returns off-diagonal elements of a matrix in flattened view
	# should be a n x n matrix
	m, n = matrix.shape
	assert m == n
	return matrix.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()


class BarlowTwins(nn.Module):

	def __init__(self, encoder, in_dim, proj_dim, regularizer):
		super(BarlowTwins, self).__init__()

		self.regularizer = regularizer
		self.encoder = encoder
		self.proj = nn.Sequential(
			nn.Linear(in_dim, in_dim),
			nn.ReLU(inplace = True),
			nn.Linear(in_dim, proj_dim))

		self.bn = nn.BatchNorm1d(proj_dim, affine = False)


 
	def forward(self, x1, x2):

		proj_1 = self.proj(self.encoder(x1)) # dim : batch_size x proj_dim
		proj_2 = self.proj(self.encoder(x2)) # dim : batch_size x proj_dim

		corr = self.bn(proj_1).T @ self.bn(proj_2)
		corr = corr.div_(proj_1.size(0)) # divide by batch size to get final cross corr matrix

		on_diag = torch.diagonal(corr).add_(-1).pow_(2).sum()
		off_diag = off_diagonal(corr).pow_(2).sum()
		loss = on_diag + self.regularizer * off_diag
		return loss




# taken from : https://github.com/facebookresearch/barlowtwins/blob/main/main.py

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



def adjust_learning_rate(optimizer, loader, step, epochs, batch_size, 
						learning_rate_weights, learning_rate_biases):

	max_steps = epochs * len(loader)
	warmup_steps = 10 * len(loader)
	base_lr = batch_size / 256

	if step > warmup_steps :
		lr = base_lr * step / warmup_steps
	else :
		step -= warmup_steps
		max_steps -= warmup_steps
		q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
		end_lr = base_lr * 0.001
		lr = base_lr * q + end_lr * (1- q)

	optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
	optimizer.param_groups[1]['lr'] = lr * learning_rate_biases

