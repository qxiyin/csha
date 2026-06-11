from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F
import logging
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict, Counter

from utils import utils
import time
import json

from utils.pomdpfl_aggregator import POMDPFLAggregator


class Server():
    def __init__(self, model, dataLoader, criterion=F.cross_entropy, device='cpu'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.pomdpFL
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""
        self.update_type = 'pomdpfl'

    def set_log_path(self, log_path, exp_name, t_run):
        self.log_path = log_path
        self.log_sim_path = '{}/sims_{}_{}.npy'.format(log_path, exp_name, t_run)
        self.log_norm_path = '{}/norms_{}_{}.npy'.format(log_path, exp_name, t_run)
        self.log_results = f'{log_path}/acc_prec_rec_f1_{exp_name}_{t_run}.txt'
        self.output_file = open(self.log_results, 'w', encoding='utf-8')

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)
        self.num_clients = len(self.clients)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        logging.info("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        nb_classes = 10
        cf_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
                for t, p in zip(target.view(-1), pred.view(-1)):
                    cf_matrix[t.long(), p.long()] += 1
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()
        logging.info(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
                test_loss, correct, count, accuracy))
        logging.info(f"[Sever] Confusion matrix:\n {cf_matrix.detach().cpu()}")
        cf_matrix = cf_matrix.detach().cpu().numpy()
        row_sum = np.sum(cf_matrix, axis=0)
        col_sum = np.sum(cf_matrix, axis=1)
        diag = np.diag(cf_matrix)
        precision = diag / row_sum
        recall = diag / col_sum
        f1 = 2 * (precision * recall) / (precision + recall)
        m_acc = np.sum(diag) / np.sum(cf_matrix)
        results = {'accuracy': accuracy, 'test_loss': test_loss,
                   'precision': precision.tolist(), 'recall': recall.tolist(),
                   'f1': f1.tolist(), 'confusion': cf_matrix.tolist(),
                   'epoch': self.iter}
        json.dump(results, self.output_file)
        self.output_file.write("\n")
        self.output_file.flush()
        logging.info(
            f"[Server] Precision={precision},\n Recall={recall},\n F1-score={f1},\n my_accuracy={m_acc * 100.}[%]")

        return test_loss, accuracy

    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update_param(self.update_type)

        if self.isSaveChanges:
            self.saveChanges(selectedClients)

        tic = time.perf_counter()
        Delta = self.AR(selectedClients)
        toc = time.perf_counter()
        total_time = toc - tic

        defense_time = getattr(self, '_last_defense_time', 0.0)

        logging.info(f"[Server] Defense time: {defense_time:0.6f} seconds")
        logging.info(f"[Server] Total aggregation process takes {total_time:0.6f} seconds.\n")

        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        logging.info(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = False
        saveOriginal = True
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            logging.info(
                f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            logging.info(f'[Server] Update vectors have been saved to {savepath}')

    def set_AR_param(self, dbscan_eps=0.5, min_samples=5):
        logging.info(f"SET DBSCAN eps={dbscan_eps}, min_samples={min_samples}")
        self.dbscan_eps = dbscan_eps
        self.min_samples = min_samples

    def set_AR(self, ar, args=None):
        if ar == 'pomdpfl':
            self.AR = self.pomdpFL
            self.update_type = 'pomdpfl'
            self.pomdpfl_args = args
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")


    def pomdpFL(self, clients):
        if not hasattr(self, 'pomdpfl_aggregator'):
            ae_retrain_interval = getattr(self.pomdpfl_args, 'ae_retrain_interval', 5) if hasattr(self,
                                                                                                  'pomdpfl_args') else 5

            self.pomdpfl_aggregator = POMDPFLAggregator(
                device=self.device,
                num_clients=len(clients),
                ae_retrain_interval=ae_retrain_interval,
                args=self.pomdpfl_args)

        if self.pomdpfl_aggregator.num_clients == 0:
            self.pomdpfl_aggregator.set_num_clients(len(clients))

        defense_start = time.perf_counter()
        valid_clients = self.pomdpfl_aggregator.get_valid_clients(clients, self.iter)
        defense_end = time.perf_counter()
        self._last_defense_time = defense_end - defense_start

        out = self.FedFuncWholeNet(valid_clients, lambda arr: self._pomdpfl_sum_with_timing(arr))
        return out

    def _pomdpfl_sum_with_timing(self, arr):
        sum_start = time.perf_counter()
        result = torch.sum(arr, dim=-1, keepdim=True)
        sum_end = time.perf_counter()
        self._last_defense_time += sum_end - sum_start
        return result

    def FedFuncWholeNet(self, clients, func):
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        sizes = [c.get_data_size() for c in clients]
        total_s = sum(sizes)
        logging.info(f"clients' sizes={sizes}, total={total_s}")
        weights = [s / total_s for s in sizes]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        weighted_vecs = [w * v for w, v in zip(weights, vecs)]
        result = func(torch.stack(weighted_vecs, 1).unsqueeze(0))
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeNetAvg(self, clients, func):
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        result = func(torch.stack(vecs, 1).unsqueeze(0))
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta

    def FedFuncWholeStateDict(self, clients, func):
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]
        resultDelta = func(deltas)
        Delta.update(resultDelta)
        return Delta