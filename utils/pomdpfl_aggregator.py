import torch
import torch.nn.functional as F
import logging
import numpy as np
from collections import defaultdict
from copy import deepcopy

from .autoencoder import AutoEncoderTrainer
from .pomdp import ContinuousPOMDP
from .supervised_detector import FineGrainedClassifier


class POMDPFLAggregator:

    def __init__(self, device='cpu', num_clients=0, tao_0=3, ae_retrain_interval=5, args=None):
        self.device = device
        self.num_clients = num_clients
        self.tao_0 = tao_0
        self.ae_retrain_interval = ae_retrain_interval
        self.args = args
        self.autoencoder_trainer = None
        self.pomdp = None
        self.fine_grained_classifier = None
        self.client_history = defaultdict(list)
        self.reconstruction_errors = defaultdict(list)
        self.abnormal_clients = set()
        self.free_rider_clients = set()
        self.permanently_banned_poisoning_clients = set()
        self.poisoning_clients = set()
        self.ae_lr = 0.0001
        self.ae_epochs = 50
        self.ae_incremental_epochs = 10
        self.anomaly_threshold_alpha = 1

        logging.info(f"[POMDP-FL] Initialized aggregator with DBSCAN-based fine-grained classification")

    def set_num_clients(self, num_clients):
        self.num_clients = num_clients
        if self.pomdp is None:
            self.pomdp = ContinuousPOMDP(num_clients, self.device)

    def get_valid_clients(self, clients, iteration):
        logging.info(f"[POMDP-FL] Starting round {iteration} with {len(clients)} clients")
        if self.permanently_banned_poisoning_clients:
            logging.info(
                f"[POMDP-FL] Permanently banned poisoning clients: {self.permanently_banned_poisoning_clients}")
            for client_idx in self.permanently_banned_poisoning_clients:
                if hasattr(clients[client_idx], 'set_detected'):
                    clients[client_idx].set_detected()

        if self.autoencoder_trainer is None:
            self._initialize_autoencoder(clients)

        if self.pomdp is None:
            self.pomdp = ContinuousPOMDP(len(clients), self.device)

        if self.fine_grained_classifier is None:
            self.fine_grained_classifier = FineGrainedClassifier(
                device=self.device,
                args=self.args
            )

        self._update_client_history(clients)

        abnormal_clients, current_reconstruction_errors = self._stage1_autoencoder_detection(clients,
                                                                                                                iteration)

        if iteration < self.tao_0:
            logging.info(
                f"[POMDP-FL] Insufficient history (round {iteration} < {self.tao_0}), using all clients except permanently banned")
            return self._filter_permanently_banned_clients(clients)

        pomdp_action = self._stage2_pomdp_decision_optimized(abnormal_clients, current_reconstruction_errors)

        if pomdp_action == 0:
            logging.info("[POMDP-FL] POMDP decision: Monitor only")
            return self._filter_permanently_banned_clients(clients)
        else:
            logging.info("[POMDP-FL] POMDP decision: Check and handle abnormal clients")
            return self._handle_abnormal_clients(clients, abnormal_clients, current_reconstruction_errors)

    def _filter_permanently_banned_clients(self, clients):
        if not self.permanently_banned_poisoning_clients:
            return clients

        valid_clients = []
        for i, client in enumerate(clients):
            if i not in self.permanently_banned_poisoning_clients:
                valid_clients.append(client)

        logging.info(
            f"[POMDP-FL] Filtered out {len(self.permanently_banned_poisoning_clients)} permanently banned clients")
        return valid_clients

    def _initialize_autoencoder(self, clients):
        sample_ema = clients[0].get_classifier_ema()

        if len(sample_ema) == 0:
            logging.warning("[POMDP-FL] No classifier parameters found, using minimal dimension")
            input_dim = 10
        else:
            input_dim = sample_ema.shape[0]

        self.autoencoder_trainer = AutoEncoderTrainer(
            input_dim=input_dim,
            device=self.device,
            lr=self.ae_lr,
            epochs=self.ae_epochs,
            incremental_epochs=self.ae_incremental_epochs,
            threshold_alpha=self.anomaly_threshold_alpha,
            retrain_interval=self.ae_retrain_interval,
            args=self.args
        )

        logging.info(f"[POMDP-FL] Initialized autoencoder with input_dim={input_dim}")

    def _update_client_history(self, clients):
        for i, client in enumerate(clients):
            if i in self.permanently_banned_poisoning_clients:
                continue

            classifier_ema = client.get_classifier_ema().detach().cpu().numpy()

            if len(classifier_ema) > 0:
                self.client_history[i].append(classifier_ema)
                if len(self.client_history[i]) > 10:
                    self.client_history[i] = self.client_history[i][-10:]
            else:
                logging.warning(f"[POMDP-FL] Client {i} has empty classifier EMA")

    def _stage1_autoencoder_detection(self, clients, current_round):
        logging.info("[POMDP-FL] Stage 1: Autoencoder-based detection (using classifier EMA)")

        training_data = []
        client_indices = []

        for i, client in enumerate(clients):
            if i in self.permanently_banned_poisoning_clients:
                continue

            classifier_ema = client.get_classifier_ema().detach().cpu().numpy()

            if len(classifier_ema) > 0:
                training_data.append(classifier_ema)
                client_indices.append(i)
            else:
                logging.warning(f"[POMDP-FL] Skipping client {i} due to empty classifier EMA")

        if len(training_data) == 0:
            logging.warning("[POMDP-FL] No valid training data for autoencoder")
            return [], {}, 0.0

        training_data = torch.FloatTensor(np.array(training_data)).to(self.device)
        self.autoencoder_trainer.train(training_data, current_round)

        abnormal_clients, reconstruction_dict = self.autoencoder_trainer.detect_anomalies(training_data, client_indices)

        logging.info(f"[POMDP-FL] Detected {len(abnormal_clients)} abnormal clients using classifier EMA")
        return abnormal_clients, reconstruction_dict

    def _stage2_pomdp_decision_optimized(self, abnormal_clients, current_reconstruction_errors):
        logging.info("[POMDP-FL] Stage 2: POMDP-based decision making")

        if len(current_reconstruction_errors) == 0:
            observation = 0.0
        else:
            abnormal_ratio = len(abnormal_clients) / max(len(current_reconstruction_errors), 1)
            errors = list(current_reconstruction_errors.values())
            avg_error = sum(errors) / len(errors)
            error_std = np.std(errors) if len(errors) > 1 else 0

            observation = min(
                abnormal_ratio * 3.0 +
                avg_error * 5000 +
                error_std * 2000,
                1.0
            )

        logging.info(f"[POMDP-FL] POMDP observation: {observation:.6f}")

        try:
            optimal_action = self.pomdp.solve_pomdp(observation)
            return optimal_action
        except Exception as e:
            logging.warning(f"[POMDP-FL] POMDP solving failed: {e}, using fallback decision")
            return 1 if observation > 0.1 else 0

    def _handle_abnormal_clients(self, clients, abnormal_clients, current_reconstruction_errors):
        logging.info("[POMDP-FL] Handling abnormal clients with DBSCAN-based fine-grained classification")

        free_rider_clients, poisoning_clients = self.fine_grained_classifier.classify_abnormal_clients(
            clients, abnormal_clients, current_reconstruction_errors
        )

        self.free_rider_clients.update(free_rider_clients)
        self.poisoning_clients = poisoning_clients

        new_poisoning_clients = poisoning_clients - self.permanently_banned_poisoning_clients
        if new_poisoning_clients:
            self.permanently_banned_poisoning_clients.update(new_poisoning_clients)
            logging.info(f"[POMDP-FL] New poisoning clients detected and permanently banned: {new_poisoning_clients}")

            for client_idx in new_poisoning_clients:
                if hasattr(clients[client_idx], 'set_detected'):
                    clients[client_idx].set_detected()

        valid_clients = []
        resubmitted_clients = []

        for i, client in enumerate(clients):
            if i in self.permanently_banned_poisoning_clients:
                logging.debug(f"[POMDP-FL] Excluding permanently banned poisoning client {i}")
                continue
            elif i in free_rider_clients:
                if hasattr(client, 'resubmit'):
                    logging.info(f"[POMDP-FL] Requesting resubmission from free-rider client {i}")
                    resubmitted_update = client.resubmit()
                    if resubmitted_update is not None:
                        client.stateChange = resubmitted_update
                        client._update_classifier_ema()
                    resubmitted_clients.append(i)
                valid_clients.append(client)
            else:
                valid_clients.append(client)

        logging.info(f"[POMDP-FL] Valid clients: {len(valid_clients)}")
        logging.info(f"[POMDP-FL] Free-rider clients: {free_rider_clients} (resubmitted: {resubmitted_clients})")
        logging.info(f"[POMDP-FL] Current round poisoning clients: {poisoning_clients}")
        logging.info(f"[POMDP-FL] Permanently banned poisoning clients: {self.permanently_banned_poisoning_clients}")

        if len(valid_clients) == 0:
            logging.warning("[POMDP-FL] No valid clients available, using all non-banned clients")
            return self._filter_permanently_banned_clients(clients)

        return valid_clients

    def reset_autoencoder_training(self):
        if self.autoencoder_trainer is not None:
            self.autoencoder_trainer.reset_training_state()
            logging.info("[POMDP-FL] Autoencoder training state reset")