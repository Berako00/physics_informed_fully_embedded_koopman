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
import openpyxl

from help_func import self_feeding, enc_self_feeding, load_model
from nn_structure import AUTOENCODER
from training import trainingfcn
from data_generation import DataGenerator, TwoLinkRobotDataGenerator
from debug_func import debug_L2, debug_L3, debug_L4, debug_L5, debug_L6
from plotting import plot_results, plot_losses, plot_debug

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

start_time = time.time()
# Data Generation

Setup = 'Twolink'# Simple, Twolink

numICs = 30000
T_step = 50
dt = 0.02

if Setup == 'Simple':
    x1range = (-0.5, 0.5)
    x2range = x1range
    mu = -0.05
    lam = -1
    seed = 1
    [train_tensor, test_tensor, val_tensor] = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)
    # NN Structure

    Num_meas = 2
    Num_inputs = 1
    Num_x_Obsv = 3
    Num_u_Obsv = 3
    Num_x_Neurons = 30
    Num_u_Neurons = 30
    Num_hidden_x_encoder = 2
    Num_hidden_x_decoder = 2
    Num_hidden_u_encoder = 2
    Num_hidden_u_decoder = 2

elif Setup == 'Twolink':
    q1_range = (-np.pi, np.pi)
    q2_range = q1_range
    dq1_range = (-6, 6)
    dq2_range = dq1_range
    [train_tensor, test_tensor, val_tensor] = TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T_step, dt)

    # NN Structure

    Num_meas = 4
    Num_inputs = 2
    Num_x_Obsv = 14
    Num_u_Obsv = 66
    Num_x_Neurons = 128
    Num_u_Neurons = 128
    Num_hidden_x_encoder = 3
    Num_hidden_u_encoder = 3

print(f"Train tensor shape: {train_tensor.shape}")
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Validation tensor shape: {val_tensor.shape}")


integrated_states = [2, 3] # The state on index 2 is being integrated for the state 0, and the state on index 3 is the derivative of the state on index 1

model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, 
                 Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, 
                 Num_hidden_u_encoder, dt, integrated_states)
# Training Loop
start_training_time = time.time()

eps = 4       # Number of epochs per batch size
lr = 1e-3        # Learning rate
batch_size = 256
S_p = 30
T = 50
alpha = [0.001, 1e-5, 1e-12]
W = 0
M = 1 # Amount of models you want to run
check_epoch = 2

[Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index, 
 Running_Losses_Array, L4_Array, L6_Array] = trainingfcn(eps, check_epoch, lr, batch_size, S_p, T, dt, integrated_states, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv,
                Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_u_encoder, train_tensor, test_tensor, M, device)

ind_loss = int(Lowest_loss_index)
L4 = np.asarray(L4_Array[ind_loss])
L6 = np.asarray(L6_Array[ind_loss])
Running_Losses = np.asarray(Running_Losses_Array[ind_loss])

# Create a dictionary with each array as a column
data = {
    "L4": L4,
    "L6": L6,
    "Running_Losses": Running_Losses
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file (all data in one sheet, different columns)
df.to_excel("training_results.xlsx", index=False)

# Load the parameters of the best model
load_model(model, Best_Model, device)
print(f"Loaded model parameters from Model: {Best_Model}")

end_time =  time.time()

total_time = end_time - start_time
total_training_time = end_time - start_training_time

print(f"Total time is: {total_time}")
print(f"Total training time is: {total_training_time}")

# Result Plotting

#plot_losses(Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array, Lowest_loss_index)
plot_debug(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
