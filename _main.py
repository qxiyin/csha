import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import json

from dataloader import Dataset_CIFAR10, Dataset_MNIST, Dataset_FMNIST
from models.init import get_model
from clients import Client
from clients_attackers import FreeRiderClient, PoisoningClient
from server import Server


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_path, exp_name, run_id):
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f'{exp_name}_{run_id}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_dataset(dataset_name, data_path='./data'):
    if dataset_name == 'CIFAR10':
        train_dataset = Dataset_CIFAR10()
        train_dataset.load(path=os.path.join(data_path, 'CIFAR10'), train=True)
        test_dataset = Dataset_CIFAR10()
        test_dataset.load(path=os.path.join(data_path, 'CIFAR10'), train=False)
    elif dataset_name == 'FMNIST':
        train_dataset = Dataset_FMNIST()
        train_dataset.load(path=os.path.join(data_path, 'FMNIST'), train=True)
        test_dataset = Dataset_FMNIST()
        test_dataset.load(path=os.path.join(data_path, 'FMNIST'), train=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset


def create_clients(args, client_datasets, device):
    clients = []

    num_abnormal = int(args.num_clients * args.attack_ratio)
    num_poisoning = int(num_abnormal * args.poisoning_ratio)
    num_freerider = num_abnormal - num_poisoning

    logging.info(f"Creating {args.num_clients} clients: "
                 f"{args.num_clients - num_abnormal} normal, "
                 f"{num_poisoning} poisoning, {num_freerider} free-rider")

    client_indices = list(range(args.num_clients))
    random.shuffle(client_indices)

    poisoning_indices = set(client_indices[:num_poisoning])
    freerider_indices = set(client_indices[num_poisoning:num_abnormal])

    for i in range(args.num_clients):
        train_loader = DataLoader(
            client_datasets[i],
            batch_size=args.batch_size,
            shuffle=True
        )

        client_model = get_model(args.model, args.dataset)
        optimizer = optim.SGD(client_model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

        attack_prob = random.uniform(args.attack_prob_min, args.attack_prob_max)

        if i in poisoning_indices:
            if args.dataset == 'CIFAR10':
                source_label, target_label = 1, 9
            elif args.dataset == 'FMNIST':
                source_label, target_label = 5, 7
            else:  # MNIST
                source_label, target_label = 1, 9

            client = PoisoningClient(
                cid=i, model=client_model, dataLoader=train_loader,
                optimizer=optimizer, device=device,
                inner_epochs=args.local_epochs,
                source_label=source_label, target_label=target_label,
                attack_prob=attack_prob
            )
        elif i in freerider_indices:
            noise_std = random.uniform(0.005, 0.01)
            client = FreeRiderClient(
                cid=i, model=client_model, dataLoader=train_loader,
                optimizer=optimizer, device=device,
                inner_epochs=args.local_epochs,
                mean=0, std=noise_std,
                attack_prob=attack_prob
            )
        else:
            client = Client(
                cid=i, model=client_model, dataLoader=train_loader,
                optimizer=optimizer, device=device,
                inner_epochs=args.local_epochs
            )

        clients.append(client)

    return clients


def main(args):
    set_seed(args.seed)
    setup_logging(args.log_path, args.exp_name, args.run_id)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    logging.info(f"Loading {args.dataset} dataset...")
    train_dataset, test_dataset = load_dataset(args.dataset)
    if args.dataset == 'CIFAR10':
        source_label, target_label = 1, 9  # Cat -> Dog
    elif args.dataset == 'FMNIST':
        source_label, target_label = 5, 7  # Sandal -> Sneaker
    else:  # MNIST
        source_label, target_label = 1, 9

    if args.attack_ratio > 0:
        logging.info(f"Creating imbalanced test set for attack evaluation...")
        logging.info(f"Attack: {source_label} -> {target_label}")

        source_test_count = getattr(args, 'source_test_count', 5000)
        target_test_count = getattr(args, 'target_test_count', 1000)
        other_test_count = getattr(args, 'other_test_count', 500)

        test_dataset = test_dataset.create_imbalanced_test_set(
            source_label=source_label,
            target_label=target_label,
            source_count=source_test_count,
            target_count=target_test_count,
            other_count=other_test_count
        )
    else:
        logging.info("No attack scenario - using balanced test set")

    is_iid = (args.iid == 'iid')
    logging.info(f"Splitting data among {args.num_clients} clients (IID: {args.iid})...")
    client_datasets = train_dataset.split(args.num_clients, iid=is_iid)

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    global_model = get_model(args.model, args.dataset)
    logging.info(f"Created {args.model} model for {args.dataset}")

    clients = create_clients(args, client_datasets, device)

    server = Server(global_model, test_loader, device=device)
    server.set_log_path(args.log_path, args.exp_name, args.run_id)

    for client in clients:
        server.attach(client)

    server.set_AR(args.defense, args)

    logging.info("=== Experiment Configuration ===")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("================================")

    logging.info("Starting federated learning...")

    for round_num in range(args.num_rounds):
        logging.info(
            f"\n================================= Round {round_num + 1}/{args.num_rounds} ================================= ")

        server.distribute()

        participating_clients = list(range(len(clients)))

        server.train(participating_clients)

        test_loss, accuracy = server.test()
        logging.info(f"Round {round_num + 1} - Test Accuracy: {accuracy:.2f}%")

    logging.info("\n=== Final Results ===")
    final_test_loss, final_accuracy = server.test()
    logging.info(f"Final Test Accuracy: {final_accuracy:.2f}%")

    results = {
        'final_accuracy': final_accuracy,
        'final_test_loss': final_test_loss,
        'experiment_config': vars(args)
    }

    results_file = os.path.join(args.log_path, f'final_results_{args.exp_name}_{args.run_id}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("Experiment completed successfully!")