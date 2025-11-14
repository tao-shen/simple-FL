from .fedavg import FedAvg
import copy
import torch
from tqdm import tqdm
import numpy as np


class FedCDA(FedAvg):
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
        self.K = 5  # Number of cached models per client
        self.B = 10  # Number of batches for approximate selection

        # Initialize model cache for each client
        self.client_model_cache = {i: [] for i in range(self.N)}

    def clients_update(self):
        """Clients train their local models and update the cache."""
        self.models = []
        for k in tqdm(self.candidates, desc="FedCDA Clients Update"):
            model = self.local_update(self.clients[k], self.server.model)
            self.models.append(model.cpu())
            # Update model cache
            self.client_model_cache[k].append(copy.deepcopy(model.cpu()))
            if len(self.client_model_cache[k]) > self.K:
                self.client_model_cache[k].pop(0)  # Maintain cache size

    def calculate_divergence(self, model1, model2):
        """Calculate the divergence between two models."""
        divergence = 0.0
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            divergence += torch.norm(param1 - param2).item()
        return divergence

    def select_models(self):
        """Select models from client caches with minimum divergence."""
        selected_models = []
        for client_id in self.candidates:
            cache = self.client_model_cache[client_id]
            if not cache:
                continue

            # If cache size is 1, just use that model
            if len(cache) == 1:
                selected_models.append(cache[0])
                continue

            # Calculate divergence for each model in cache
            divergences = []
            current_model = cache[-1]  # Latest model
            
            # Calculate divergence between current model and each cached model
            for cached_model in cache[:-1]:  # Exclude current model
                div = self.calculate_divergence(current_model, cached_model)
                divergences.append(div)

            # Select model with minimum divergence
            min_div_idx = np.argmin(divergences)
            selected_models.append(cache[min_div_idx])

        return selected_models

    def server_update(self):
        """Update server model using selected cross-round models."""
        selected_models = self.select_models()
        if not selected_models:  # If no models were selected, use the current round models
            selected_models = self.models
        w = self.averaging(selected_models)
        self.server.model.load_state_dict(w)

    def training(self):
        for t in range(self.T):
            print(f"========== Communication Round {t} ==========")
            self.candidates_sampling()
            self.clients_update()
            self.server_update()
            self.evaluate() 