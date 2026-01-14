from simplefl.methods.fedavg import FedAvg
from .fl import *
from simplefl.utils import *
import numpy as np
from tqdm import tqdm
import copy
import torch


class FedSpeed(FedAvg):
    
    def __init__(self, server, clients, args):
        super().__init__(server, clients, args)
    
    def local_update(self, client, model_g):
        model_l = copy.deepcopy(model_g)
        optimizer = self.opts[self.args.local_optimizer](
            model_l.parameters(), lr=self.args.lr_l, weight_decay=self.args.weight_decay)
        try:
            train_loader = DataLoader(
                Dataset(client.train_data, self.args), batch_size=self.args.local_batch_size, shuffle=True)
        except:  # skip dataless client
            return model_l
        
        # Initialize grad_prox at the beginning of each communication round
        # This should be reset each round, not just the first time
        client.grad_prox = {name: torch.zeros_like(param).cpu() for name, param in model_l.named_parameters()}

        for E in range(self.E):
            model_l = self.fit_fedspeed(model_l, train_loader, optimizer, model_g, client)
        return model_l

    def fit_fedspeed(self, model, train_loader, optimizer, model_g, client, alpha=0.9, rho=0.001, lambda_=100):
        """
        FedSpeed local training step according to the paper:
        "FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy"
        
        Key steps:
        1. Compute gradient at current point: ∇f(w)
        2. Compute perturbed point: w' = w + ρ * ∇f(w) (gradient ascent)
        3. Compute gradient at perturbed point: ∇f(w')
        4. Combine gradients: g̃ = (1-α) * ∇f(w) + α * ∇f(w')
        5. Apply prox-correction: g = g̃ - ĝ_{t-1} + (1/λ) * (w - w_g)
        6. Update parameters: w_{t+1} = w_t - η * g
        7. Update prox-correction term: ĝ_t = ĝ_{t-1} - (1/λ) * (w_{t+1} - w_g)
        """
        model.train().to(model.device)
        model_g_dict = model_g.to(model.device).state_dict()
        
        epochs = tqdm(train_loader, leave=False, desc='local_update')
        for idx, batch in enumerate(epochs):
            optimizer.zero_grad()
            batch = to_device(batch, model.device)
            
            # Step 1: Compute gradient at current point ∇f(w)
            output = model(batch)
            label = batch['label']
            loss = model.loss_fn(output, label)
            loss.backward()
            grad_1 = {name: param.grad.data.clone() for name, param in model.named_parameters()}
            
            # Step 2 & 3: Compute gradient at perturbed point (only if rho > 0 and alpha > 0)
            if rho > 0 and alpha > 0:
                # Step 2: Compute perturbed point w' = w + ρ * ∇f(w) (gradient ascent for perturbation)
                with torch.no_grad():
                    perturbed_state = {name: param.data + rho * param.grad.data 
                                       for name, param in model.named_parameters()}
                
                optimizer.zero_grad()
                
                # Step 3: Compute gradient at perturbed point ∇f(w')
                model_perturbed = copy.deepcopy(model)
                model_perturbed.load_state_dict(perturbed_state)
                model_perturbed.train().to(model.device)
                output_perturbed = model_perturbed(batch)
                loss_perturbed = model_perturbed.loss_fn(output_perturbed, label)
                loss_perturbed.backward()
                grad_2 = {name: param.grad.data.clone() for name, param in model_perturbed.named_parameters()}
            else:
                # When rho=0 or alpha=0, grad_2 = grad_1 (no perturbation)
                grad_2 = grad_1
            
            optimizer.zero_grad()
            
            # Step 4: Combine gradients: g̃ = (1-α) * ∇f(w) + α * ∇f(w')
            # Step 5: Apply prox-correction: g = g̃ - ĝ_{t-1} + (1/λ) * (w - w_g)
            for n, p in model.named_parameters():
                # Combine gradients
                grad_combined = (1 - alpha) * grad_1[n] + alpha * grad_2[n]
                
                # Apply prox-correction term (only if lambda_ is finite and positive)
                if lambda_ < float('inf') and lambda_ > 0:
                    grad_prox_term = client.grad_prox[n].to(p.device)
                    prox_term = (1.0 / lambda_) * (p.data - model_g_dict[n])
                    final_grad = grad_combined - grad_prox_term + prox_term
                else:
                    # When lambda_ is infinite or zero, no prox correction (pure FedAvg)
                    final_grad = grad_combined
                
                # Set gradient for optimizer
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                p.grad.data = final_grad
            
            # Step 6: Update parameters: w_{t+1} = w_t - η * g
            optimizer.step()
        
        # Step 7: Update prox-correction term after all batches in this epoch
        # ĝ_t = ĝ_{t-1} - (1/λ) * (w_{t+1} - w_g)
        # Only update if lambda_ is finite and positive
        if lambda_ < float('inf') and lambda_ > 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    delta = p.data - model_g_dict[n]
                    client.grad_prox[n] = client.grad_prox[n].cpu() - (1.0 / lambda_) * delta.cpu()
        
        return model
