from __future__ import print_function

from copy import deepcopy
from collections import deque

import torch
import torch.nn.functional as F
import logging
import numpy as np

from utils import utils


class Client():
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.cross_entropy, device='cpu', inner_epochs=1):
        self.cid = cid
        self.model = model
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.device = device
        self.log_interval = len(dataLoader) - 1
        self.init_stateChange()
        self.originalState = deepcopy(model.state_dict())
        self.isTrained = False
        self.inner_epochs = inner_epochs
        self.criterion = criterion
        self.K_avg = 3
        self.hog_avg = deque(maxlen=self.K_avg)
        self.classifier_delta = {}
        self.ema_alpha = 0.2
        self.classifier_ema = None
        self.is_ema_initialized = False

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states
        self.avg_delta = deepcopy(states)
        self.sum_hog = deepcopy(states)

    def setModelParameter(self, states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()

    def data_transform(self, data, target):
        return data, target

    def get_data_size(self):
        return len(self.dataLoader.dataset)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = self.data_transform(data, target)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.isTrained = True
        self.model.cpu()

    def test(self, testDataLoader):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testDataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testDataLoader.dataset)
        self.model.cpu()
        logging.info('client {} ## Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            self.cid, test_loss, correct, len(testDataLoader.dataset),
            100. * correct / len(testDataLoader.dataset)))

    def update_param(self, update_type='common'):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()

        if update_type == "pomdpfl":
            for p in self.originalState:
                diff = newState[p] - self.originalState[p]
                self.stateChange[p] = diff

            self._update_classifier_ema()

        self.isTrained = False

    def _update_classifier_ema(self):
        self.classifier_delta = {}
        classifier_params = utils.getClassifierParameters(self.model)

        for param in classifier_params:
            if param in self.stateChange:
                self.classifier_delta[param] = self.stateChange[param]

        current_classifier_vec = torch.cat([v.flatten().float() for v in self.classifier_delta.values()])

        if not self.is_ema_initialized:
            self.classifier_ema = current_classifier_vec.clone()
            self.is_ema_initialized = True
        else:
            self.classifier_ema = self.ema_alpha * self.classifier_ema + (1 - self.ema_alpha) * current_classifier_vec

    def getDelta(self):
        return self.stateChange

    def get_avg_grad(self):
        return torch.cat([v.flatten() for v in self.avg_delta.values()])

    def get_sum_hog(self):
        return torch.cat([v.flatten() for v in self.sum_hog.values()])

    def get_L2_sum_hog(self):
        X = self.get_sum_hog()
        return torch.linalg.norm(X)

    def get_L2_avg_grad(self):
        X = torch.cat([v.flatten() for v in self.avg_delta.values()])
        return torch.linalg.norm(X)

    def get_L2_last_grad(self):
        X = torch.cat([v.flatten() for v in self.stateChange.values()])
        return torch.linalg.norm(X)

    def getClassifierDelta(self):
        return self.classifier_delta

    def get_classifier_ema(self):
        return self.classifier_ema

    def resubmit(self):
        return self.stateChange