import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as r
import time
import math
import openpyxl
import json

from help_func import self_feeding, enc_self_feeding, load_model
from nn_structure import AUTOENCODER
from training import trainingfcn
from data_generation import DataGenerator, TwoLinkRobotDataGenerator

from ga_optimizer import run_genetic_algorithm
import multiprocessing as mp

def main():
    mp.set_start_method("spawn", force=True)
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    start_time = time.time()
    
    # ---- System Params ----------
    
    system = 'two_link'     # 'two_link' or 'simple'
    # ----------------------------
    
    # ------- Data Generation Params ----------
    numICs = 30000
    T_step = 50
    dt = 0.02
    seed = 1
    
    # ---- GA Params -------------
    use_ga = True
    generations = 2
    pop_size = 5
    eps = 5
    tournament_size = 3
    mutation_rate = 0.2
    
    # Define parameter ranges For GA
    param_ranges = {
        "Num_x_Obsv": (8, 100),
        "Num_u_Obsv": (8, 100),
        "Num_x_Neurons": (128, 128),
        "Num_u_Neurons": (128, 128),
        "Num_hidden_x": (3, 3),  # Shared for both x encoder and decoder
        "Num_hidden_u": (3, 3),  # Shared for both u encoder and decoder
        "alpha0": (0.1, 0.1),
        "alpha1": (1e-9, 1e-5),
        "alpha2": (1e-18, 1e-12)
    }
    # ------------------------------
    
    if system == 'simple':
      x1range = (-0.5, 0.5)
      x2range = x1range
      mu = -0.05
      lam = -1
    
      Num_meas = 2
      Num_inputs = 1
    
      [train_tensor, test_tensor, val_tensor] = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)
    
      if not use_ga:
          Num_x_Obsv    = 9
          Num_u_Obsv    = 4
          Num_x_Neurons = 128
          Num_u_Neurons = 128
          Num_hidden_x  = 3
          Num_hidden_u  = 3
          alpha         = [0.001, 1e-5, 1e-17]
    
      
    elif system == 'two_link':
      q1_range = (-math.pi, math.pi)
      q2_range = (-math.pi, math.pi)
      dq1_range = (-6, 6)
      dq2_range = dq1_range
      tau_max = 7.5
    
      Num_meas = 4
      Num_inputs = 2
    
      [train_tensor, test_tensor, val_tensor] = TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T_step, dt, tau_max)
    
      if not use_ga:
            Num_x_Obsv    = 29
            Num_u_Obsv    = 48
            Num_x_Neurons = 128
            Num_u_Neurons = 128
            Num_hidden_x  = 3
            Num_hidden_u  = 3
            alpha         = [0.001, 1e-5, 1e-14]
    
    
    # ---- Define last training param -------
    eps_final = 10      # Number of epochs for final training
    breakout = 10
    check_epoch = 1
    lr = 1e-3       # Learning rate
    batch_size = 256
    S_p = 30
    T = len(train_tensor[0, :, :])
    W = 0
    M = 1  # Amount of models you want to run
    # ---------------------------------------
    
    
    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Test tensor shape: {test_tensor.shape}")
    print(f"Validation tensor shape: {val_tensor.shape}")
    
    # --- Genetic Algorithm Hyperparameter Optimization ---
    if use_ga:
        # For speed, use a lower number of epochs for evaluation (eps) and fewer generations/population size.
        best_params = run_genetic_algorithm(check_epoch, Num_meas, Num_inputs, train_tensor, test_tensor, tournament_size, mutation_rate, generations, pop_size, eps, lr, batch_size, S_p, T, dt, param_ranges=param_ranges, elitism_count=1)
    
        Num_meas      = best_params['Num_meas']
        Num_inputs    = best_params['Num_inputs']
        Num_x_Obsv    = best_params['Num_x_Obsv']
        Num_u_Obsv    = best_params['Num_u_Obsv']
        Num_x_Neurons = best_params['Num_x_Neurons']
        Num_u_Neurons = best_params['Num_u_Neurons']
        Num_hidden_x  = best_params['Num_hidden_x']
        Num_hidden_u  = best_params['Num_hidden_u']
        alpha         = [best_params['alpha0'], best_params['alpha1'], best_params['alpha2']]
    
        # --- SAVE BEST PARAMS TO TEXT FILE ---
        with open("best_params.txt", "w") as f:
            json.dump(best_params, f, indent=4)
        print("Saved GA best parameters to best_params.txt")

if __name__ == "__main__":
    main()
