from __future__ import print_function

import torch
import torch.nn.functional as F
import logging
import random
from copy import deepcopy

from utils import utils
from clients import *


class FreeRiderClient(Client):
    def __init__(self, cid, model, dataLoader, optimizer,
                 criterion=F.cross_entropy, device='cpu', inner_epochs=1,
                 mean=0.0, std=0.01, attack_prob=0.9):
        super(FreeRiderClient, self).__init__(cid, model, dataLoader,
                                              optimizer, criterion,
                                              device, inner_epochs)
        self.mean = mean
        self.std = std
        self.attack_prob = attack_prob
        self.is_attacking = False

    def decide_attack(self):
        return self.is_attacking

    def update_param(self, update_type='pomdpfl'):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'

        was_attacking = self.is_attacking
        self.is_attacking = (random.random() < self.attack_prob)
        if self.is_attacking and not was_attacking:
            logging.info(f"FreeRiderClient {self.cid} initiated attack (this round)")
        elif not self.is_attacking and was_attacking:
            logging.info(f"FreeRiderClient {self.cid} stopped attacking (this round)")

        if not self.is_attacking:
            super().update_param(update_type)
        else:
            newState = self.model.state_dict()
            self.raw_stateChange = {}

            if update_type == "pomdpfl":
                trainable_parameter = utils.getTrainableParameters(self.model)
                for p in self.originalState:
                    diff = newState[p] - self.originalState[p]
                    self.stateChange[p] = diff
                    self.raw_stateChange[p] = diff.clone()
                    if p not in trainable_parameter:
                        continue
                    std = torch.ones(self.stateChange[p].shape) * self.std
                    noise = torch.normal(mean=self.mean, std=std)
                    self.stateChange[p] = noise

                self._update_classifier_ema()

            self.isTrained = False

    def resubmit(self):
        self.is_attacking = False
        logging.info(f"FreeRiderClient {self.cid} was detected and resubmitted valid model")
        return self.raw_stateChange


class PoisoningClient(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.cross_entropy,
                 device='cpu', inner_epochs=1, source_label=9, target_label=7, attack_prob=0.9):
        super(PoisoningClient, self).__init__(cid, model, dataLoader, optimizer,
                                              criterion, device, inner_epochs)
        self.source_label = source_label
        self.target_label = target_label
        self.attack_prob = attack_prob
        self.is_attacking = False
        self.detected = False
        logging.info(
            f"Initialized Poisoning Client {cid} with label flipping {source_label} -> {target_label}, attack_prob={attack_prob}")

    def decide_attack(self):
        return self.is_attacking

    def data_transform(self, data, target):
        if self.detected:
            return data, target

        if self.is_attacking:
            target_ = torch.tensor(list(map(lambda x: self.target_label if x == self.source_label else x, target)))
            return data, target_
        else:
            return data, target

    def train(self):
        if self.detected:
            logging.info(f"PoisoningClient {self.cid} is permanently banned, skipping training")
            self.isTrained = True
            return

        was_attacking = self.is_attacking
        self.is_attacking = (random.random() < self.attack_prob)
        if self.is_attacking and not was_attacking:
            logging.info(f"PoisoningClient {self.cid} initiated attack (this round)")
        elif not self.is_attacking and was_attacking:
            logging.info(f"PoisoningClient {self.cid} stopped attacking (this round)")

        super().train()

    def set_detected(self):
        self.is_attacking = False
        self.detected = True
        logging.info(f"PoisoningClient {self.cid} was detected and permanently blocked from training")
