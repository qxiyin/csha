import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super(AutoEncoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 4),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        logging.info("[AutoEncoder] Encoder parameters frozen")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        logging.info("[AutoEncoder] Encoder parameters unfrozen")

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        logging.info("[AutoEncoder] Decoder parameters frozen")

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True
        logging.info("[AutoEncoder] Decoder parameters unfrozen")


class AutoEncoderTrainer:


    def __init__(self, input_dim, device='cpu', lr=0.001, epochs=50, threshold_alpha=1.0,
                 retrain_interval=5, incremental_epochs=10, args=None):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.incremental_epochs = incremental_epochs
        self.threshold_alpha = threshold_alpha
        self.retrain_interval = retrain_interval

        if args is not None:
            self.threshold_multiplier = getattr(args, 'threshold_multiplier', 1)
            self.max_threshold_alpha = getattr(args, 'max_threshold_alpha', 1.5)
            self.threshold_adaptation_rate = getattr(args, 'threshold_adaptation_rate', 0.1)
        else:
            self.threshold_multiplier = 1
            self.max_threshold_alpha = 1.5
            self.threshold_adaptation_rate = 0.1

        self.autoencoder = AutoEncoder(input_dim).to(device)

        self.full_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(self.autoencoder.decoder.parameters(), lr=lr)

        self.is_trained = False
        self.last_retrain_round = -1
        self.training_history = []
        self.max_history_size = 1000
        self.patience = 5
        self.min_delta = 1e-6
        self.base_threshold_alpha = threshold_alpha
        self.min_threshold_alpha = 1
        self.error_history = []
        self.stability_window = 10

        logging.info(f"[AutoEncoder] Initialized with input_dim={input_dim}, retrain_interval={retrain_interval}")
        logging.info(f"[AutoEncoder] Hyperparameters - threshold_multiplier={self.threshold_multiplier}, "
                     f"max_threshold_alpha={self.max_threshold_alpha}, "
                     f"threshold_adaptation_rate={self.threshold_adaptation_rate}")

    def should_retrain(self, current_round):
        if not self.is_trained:
            return True
        if current_round - self.last_retrain_round >= self.retrain_interval:
            return True
        return False

    def add_training_data(self, data):
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.array(data)

        self.training_history.append(data_np)

        if len(self.training_history) > self.max_history_size:
            self.training_history = self.training_history[-self.max_history_size:]

    def get_combined_training_data(self, current_data):
        if len(self.training_history) == 0:
            return current_data

        combined_data = []
        for hist_data in self.training_history:
            combined_data.append(hist_data)

        if isinstance(current_data, torch.Tensor):
            combined_data.append(current_data.detach().cpu().numpy())
        else:
            combined_data.append(np.array(current_data))

        combined_np = np.vstack(combined_data)
        return torch.FloatTensor(combined_np).to(self.device)

    def train_with_early_stopping(self, training_data, max_epochs, training_type="", use_decoder_only=False):
        self.autoencoder.train()
        best_loss = float('inf')
        patience_counter = 0

        if use_decoder_only:
            optimizer = self.decoder_optimizer
            logging.info("[AutoEncoder] Using decoder-only optimizer for incremental training")
        else:
            optimizer = self.full_optimizer
            logging.info("[AutoEncoder] Using full optimizer for complete training")

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            reconstructed = self.autoencoder(training_data)
            loss = F.mse_loss(reconstructed, training_data)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss - self.min_delta:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logging.info(f"[AutoEncoder] {training_type} Early stopping at epoch {epoch}, loss: {loss.item():.6f}")
                break

            if epoch % 10 == 0:
                logging.debug(f"[AutoEncoder] {training_type} epoch {epoch}, loss: {loss.item():.6f}")

        final_loss = loss.item()
        logging.info(f"[AutoEncoder] {training_type} completed, final loss: {final_loss:.6f}")
        return final_loss

    def train_full(self, training_data, current_round):
        logging.info(f"[AutoEncoder] Full retraining at round {current_round}")

        self.autoencoder.unfreeze_encoder()
        self.autoencoder.unfreeze_decoder()

        self.full_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)

        full_training_data = self.get_combined_training_data(training_data)

        self.train_with_early_stopping(
            full_training_data,
            self.epochs,
            "Full training",
            use_decoder_only=False
        )

        self.is_trained = True
        self.last_retrain_round = current_round

    def train_incremental(self, training_data):
        logging.info("[AutoEncoder] Incremental training (encoder frozen)")

        self.autoencoder.freeze_encoder()
        self.autoencoder.unfreeze_decoder()

        self.decoder_optimizer = torch.optim.Adam(
            self.autoencoder.decoder.parameters(),
            lr=self.lr
        )

        self.train_with_early_stopping(
            training_data,
            self.incremental_epochs,
            "Incremental training (decoder only)",
            use_decoder_only=True
        )

    def train(self, training_data, current_round=0):
        self.add_training_data(training_data)

        if self.should_retrain(current_round):
            self.train_full(training_data, current_round)
        else:
            self.train_incremental(training_data)

    def _adapt_threshold(self, errors):
        if len(errors) < 5:
            return self.threshold_alpha

        error_std = np.std(errors)
        error_mean = np.mean(errors)

        self.error_history.append(error_mean)
        if len(self.error_history) > self.stability_window:
            self.error_history = self.error_history[-self.stability_window:]

        if len(self.error_history) >= 5:
            recent_stability = np.std(self.error_history[-5:])
            if error_mean < 0.001 and recent_stability < 0.0001:
                new_alpha = min(self.max_threshold_alpha,
                                self.threshold_alpha + self.threshold_adaptation_rate)
                logging.info(f"[AutoEncoder] Adapting threshold UP: {self.threshold_alpha:.2f} -> {new_alpha:.2f}")
                self.threshold_alpha = new_alpha
            elif error_std > error_mean * 0.5:
                new_alpha = max(self.min_threshold_alpha,
                                self.threshold_alpha - self.threshold_adaptation_rate)
                logging.info(f"[AutoEncoder] Adapting threshold DOWN: {self.threshold_alpha:.2f} -> {new_alpha:.2f}")
                self.threshold_alpha = new_alpha

        return self.threshold_alpha

    def detect_anomalies(self, data, client_indices):
        self.autoencoder.eval()

        with torch.no_grad():
            reconstructed = self.autoencoder(data)
            reconstruction_errors = torch.mean((data - reconstructed) ** 2, dim=1)

        errors = reconstruction_errors.cpu().numpy()
        current_alpha = self._adapt_threshold(errors)

        mean_error = np.mean(errors)
        std_error = np.std(errors)
        q75 = np.percentile(errors, 75)
        q25 = np.percentile(errors, 25)
        iqr = q75 - q25
        outlier_threshold = q75 + 1.5 * iqr
        statistical_threshold = mean_error + current_alpha * std_error
        threshold = max(statistical_threshold, outlier_threshold)
        threshold = threshold * self.threshold_multiplier

        logging.info(f"[AutoEncoder] Reconstruction threshold: {threshold:.6f} "
                     f"(alpha: {current_alpha:.2f}, mean: {mean_error:.6f}, std: {std_error:.6f}, "
                     f"multiplier: {self.threshold_multiplier})")

        abnormal_clients = []
        reconstruction_dict = {}

        for i, error in enumerate(reconstruction_errors):
            client_idx = client_indices[i]
            reconstruction_dict[client_idx] = error.item()

            if error.item() > threshold:
                abnormal_clients.append(client_idx)
                logging.info(f"[AutoEncoder] Client {client_idx} flagged as abnormal (error: {error.item():.6f})")

        return abnormal_clients, reconstruction_dict

    def reset_training_state(self):
        self.is_trained = False
        self.last_retrain_round = -1
        self.training_history = []
        self.error_history = []
        self.threshold_alpha = self.base_threshold_alpha
        self.autoencoder.unfreeze_encoder()
        self.autoencoder.unfreeze_decoder()
        self.full_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        self.decoder_optimizer = torch.optim.Adam(self.autoencoder.decoder.parameters(), lr=self.lr)

        logging.info("[AutoEncoder] Training state reset, all parameters unfrozen")

    def get_training_status(self):
        encoder_frozen = not any(param.requires_grad for param in self.autoencoder.encoder.parameters())
        decoder_frozen = not any(param.requires_grad for param in self.autoencoder.decoder.parameters())

        status = {
            'is_trained': self.is_trained,
            'last_retrain_round': self.last_retrain_round,
            'encoder_frozen': encoder_frozen,
            'decoder_frozen': decoder_frozen,
            'training_history_size': len(self.training_history),
            'current_threshold_alpha': self.threshold_alpha
        }

        return status

    def log_training_status(self):
        status = self.get_training_status()
        logging.info(f"[AutoEncoder] Training Status: {status}")