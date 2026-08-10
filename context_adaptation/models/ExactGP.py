#!/usr/bin/env python3
import numpy as np
import os
import yaml

import gpytorch
gpytorch.settings.debug(False)

import torch
import torch.nn.functional as F

from scipy import stats


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,lengthscale):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = lengthscale


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class GPTraversability():
    def __init__(self, feats, labels, config, params_dir = None):

        self.lengthscale = config['lengthscale']
        self.update_thresh = config['update_thresh']
        self.train_kernel = config['train_kernel']
        self.prediction_batch_size = config.get('prediction_batch_size', 4096)

        if self.train_kernel:
            print("GP KERNEL TRAINING ENABLED, maybe no a good idea online")
        
        self.gp_params = None
        self.params_dir = params_dir

        if self.params_dir is not None:
            self.load_params(self.params_dir)

        self.in_mean = torch.mean(feats, dim = 0)
        self.in_std = torch.std(feats, dim = 0) + 1e-6
        self.out_mean = torch.mean(labels)
        self.out_std = torch.std(labels) + 1e-6

        feats = (feats - self.in_mean)/self.in_std
        labels = (labels - self.out_mean)/self.out_std

        self.feats = feats
        self.labels = labels
        self.init_size = len(self.feats)

        self.reinit_model()
        
    def reinit_model(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.gp = ExactGPModel(self.feats, self.labels, self.likelihood, self.lengthscale).cuda()

        if self.gp_params is not None:
            self.gp.load_state_dict(self.gp_params)


    def load_params(self, params_dir):
        self.gp_params = torch.load(params_dir, weights_only = True)
        

    def train(self, feats, labels):
        if (len(feats) - len(self.feats) > self.update_thresh) or (len(self.feats) == self.init_size):
            self.in_mean = torch.mean(feats, dim = 0)
            self.in_std = torch.std(feats, dim = 0)

            self.out_mean = torch.mean(labels)
            self.out_std = torch.std(labels)

            feats = (feats - self.in_mean)/self.in_std
            labels = (labels - self.out_mean)/self.out_std
            
            self.feats = feats
            self.labels = labels

            self.reinit_model()

        if self.train_kernel and len(feats) > 15:
            optimizer = torch.optim.SGD(self.gp.parameters(), lr=0.01)
            with gpytorch.settings.debug(False):
                self.gp.train()
                self.likelihood.train()
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
                optimizer.zero_grad()
                output = self.gp(feats)
                # print(output)
                loss = -mll(output, labels)
                print("LOSS - ", loss.item())
                # print("PARAMS ", self.gp.covar_module)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
            
                self.gp_params = self.gp.state_dict()

    def forward(self, feats, cvar = None):
        self.gp.eval()
        self.likelihood.eval()

        feats = (feats - self.in_mean)/self.in_std

        predictions = []
        variances = []
        for batch in feats.split(self.prediction_batch_size):
            gp_pred = self.likelihood(self.gp(batch))
            predictions.append(gp_pred.mean)
            variances.append(gp_pred.variance)
        pred = torch.cat(predictions)
        var = torch.cat(variances)

        if cvar is not None:
            phi = stats.norm.pdf(stats.norm.ppf(cvar))
            pred = pred + (var * phi)/(1.0-cvar)

        pred = (pred * self.out_std) + self.out_mean

        return pred, var
