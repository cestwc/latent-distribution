import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

class GaussianKernelDensity(nn.Module):
	def __init__(self, X, bandwidth=1.0):
		super(GaussianKernelDensity, self).__init__()
		self.bandwidth = nn.Parameter(torch.tensor(bandwidth))
		self.X = torch.nn.Parameter(torch.as_tensor(X))
		n, d = X.shape
		self.coeff = - math.log(n) - math.log(2 * math.pi)  * d / 2  - math.log(self.bandwidth) * d
	
	def forward(self, x):
		return torch.cdist(x, self.X, p=2).pow(2).div(-2*self.bandwidth**2).logsumexp(1) + self.coeff
	
	def sample(self, n_samples=1):
		with torch.no_grad():
			indices = torch.randint(0, len(self.X), (n_samples,))
			return torch.normal(self.X[indices], self.bandwidth)

class kernelLogLikelihood(nn.Module):
	def __init__(self, X, n_components=100, bandwidth = np.logspace(-1, 3, 200)):
		super(kernelLogLikelihood, self).__init__()
	
		pca = PCA(n_components=n_components, whiten=False)
		X_pca = pca.fit_transform(X)
		self.pca = nn.Linear(*pca.components_.T.shape)
		self.pca_inverse = nn.Linear(*pca.components_.shape)
	
		with torch.no_grad():
			self.pca.weight.copy_(torch.as_tensor(pca.components_))
			self.pca_inverse.weight = nn.Parameter(self.pca.weight.T)
			self.pca_inverse.bias.copy_(torch.as_tensor(X).mean(dim = 0))
			self.pca.bias = nn.Parameter(-torch.matmul(self.pca_inverse.bias, self.pca.weight.T))
	
		if bandwidth is not None:
			grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidth}, n_jobs=-1)
			grid.fit(X_pca)
			bandwidth = grid.best_estimator_.bandwidth
		else:
			bandwidth = 1.0
		self.kde = GaussianKernelDensity(torch.as_tensor(X_pca), bandwidth = bandwidth)
	
	def forward(self, x):
		return self.kde(self.pca(x))
	
	def sample(self, n_samples=1):
		X = self.kde.sample(n_samples)
		return self.pca_inverse(X)

class KernelDistribution(nn.Module):
	def __init__(self, *X_, n_components=15, bandwidth = np.logspace(-1, 3, 200), prior = []):
		super(KernelDistribution, self).__init__()
		self.loglikelihood = nn.ModuleList([])
		for i, X in enumerate(X_):
			self.loglikelihood.append(kernelLogLikelihood(X, n_components, bandwidth))
			prior.append(len(X))
	
			self.prior = nn.Parameter(torch.tensor(prior[:len(X_)]) / sum(prior[:len(X_)]))
	
	def forward(self, x):
		futures = [torch.jit.fork(k, x) for k in self.loglikelihood]
		loglikelihood = [torch.jit.wait(future) for future in futures]
		loglikelihood = torch.vstack(loglikelihood).T
		logjoint = loglikelihood + self.prior.log()
		return logjoint
	
	def sample(self, n_samples=2):
		counts = np.random.multinomial(n_samples, self.prior.tolist())
		futures = [torch.jit.fork(k.sample, n) for k, n in zip(self.loglikelihood, counts)]
		X = torch.vstack([torch.jit.wait(future) for future in futures])
		y = torch.arange(len(counts)).repeat_interleave(torch.from_numpy(counts))
		return X, y

class ConvolvedDistribution(nn.Module):
	def __init__(self, distribution, func_pre, func_post):
		super(ConvolvedDistribution, self).__init__()
		self.original = distribution
		self.func_pre = func_pre
		self.func_post = func_post
	
	def forward(self, x):
		_x = self.func_pre(x)
		_logjoint = self.original(_x)
		return self.func_post(_logjoint)
