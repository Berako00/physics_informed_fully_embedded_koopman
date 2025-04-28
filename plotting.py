import random
import matplotlib
import torch
matplotlib.use("TkAgg")  # Enables interactive plotting
import matplotlib.pyplot as plt

from debug_func import debug_L1, debug_L2, debug_L3, debug_L4, debug_L5, debug_L6
from help_func import self_feeding, enc_self_feeding

def plot_losses(Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array, Lowest_test_loss_index):

    # Set fontsizes for title, labels, and legend.
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # Retrieve default color cycle from matplotlib.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Increase figure size to accommodate the additional subplots.
    fig = plt.figure(figsize=(18, 8))

    # Convert index to integer.
    idx = int(Lowest_test_loss_index)

    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgx_Array[idx], label='Lgx', color=colors[0])
    ax_top.plot(Lgu_Array[idx], label='Lgu', color=colors[1])
    ax_top.plot(L3_Array[idx], label='L3', color=colors[2])
    ax_top.plot(L4_Array[idx], label='L4', color=colors[3])
    ax_top.plot(L5_Array[idx], label='L5', color=colors[4])
    ax_top.plot(L6_Array[idx], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses', fontsize=title_fontsize)
    ax_top.set_yscale('log')  # Set y-axis to logarithmic scale

    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgx_Array[idx], label='Lgx', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgx', fontsize=title_fontsize)
    ax1.set_yscale('log')

    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgu_Array[idx], label='Lgu', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgu', fontsize=title_fontsize)
    ax2.set_yscale('log')

    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_Array[idx], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)
    ax3.set_yscale('log')

    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_Array[idx], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)
    ax4.set_yscale('log')

    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_Array[idx], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)
    ax5.set_yscale('log')

    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_Array[idx], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)
    ax6.set_yscale('log')

    plt.tight_layout()
    plt.show()


def plot_losses_mixed(Lgx_unforced_Array, Lgu_forced_Array, L3_forced_Array, L4_forced_Array, L5_forced_Array, L6_forced_Array,
                      L3_unforced_Array, L4_unforced_Array, L5_unforced_Array, L6_unforced_Array, Lowest_loss_index):
    # Set fontsizes for title, labels, and legend.
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # Retrieve default color cycle from matplotlib.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    idx = int(Lowest_loss_index)

    fig = plt.figure(figsize=(18, 8))

    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgx_unforced_Array[idx], label='Lgx', color=colors[0])
    ax_top.plot(L3_unforced_Array[idx], label='L3', color=colors[2])
    ax_top.plot(L4_unforced_Array[idx], label='L4', color=colors[3])
    ax_top.plot(L5_unforced_Array[idx], label='L5', color=colors[4])
    ax_top.plot(L6_unforced_Array[idx], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses Unforced', fontsize=title_fontsize)

    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgx_unforced_Array[idx], label='Lgx', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgx', fontsize=title_fontsize)

    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgx_unforced_Array[idx], label='Lgx', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgx', fontsize=title_fontsize)

    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_unforced_Array[idx], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)

    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_unforced_Array[idx], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)

    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_unforced_Array[idx], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)

    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_unforced_Array[idx], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(18, 8))

    # Top subplot: spans both columns of a 4x2 grid (first row)
    ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax_top.plot(Lgu_forced_Array[idx], label='Lgu', color=colors[0])
    ax_top.plot(L3_forced_Array[idx], label='L3', color=colors[2])
    ax_top.plot(L4_forced_Array[idx], label='L4', color=colors[3])
    ax_top.plot(L5_forced_Array[idx], label='L5', color=colors[4])
    ax_top.plot(L6_forced_Array[idx], label='L6', color=colors[5])
    ax_top.legend(loc='upper right', fontsize=legend_fontsize)
    ax_top.set_xlabel('Epochs', fontsize=label_fontsize)
    ax_top.set_ylabel('Loss', fontsize=label_fontsize)
    ax_top.set_title('Losses Forced', fontsize=title_fontsize)

    # Lower subplots: arranged as a 3x2 grid.
    ax1 = plt.subplot2grid((4, 2), (1, 0))
    ax1.plot(Lgu_forced_Array[idx], label='Lgu', color=colors[0])
    ax1.legend(loc='upper right', fontsize=legend_fontsize)
    ax1.set_xlabel('Epochs', fontsize=label_fontsize)
    ax1.set_ylabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Lgu', fontsize=title_fontsize)

    ax2 = plt.subplot2grid((4, 2), (1, 1))
    ax2.plot(Lgu_forced_Array[idx], label='Lgu', color=colors[1])
    ax2.legend(loc='upper right', fontsize=legend_fontsize)
    ax2.set_xlabel('Epochs', fontsize=label_fontsize)
    ax2.set_ylabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Lgu', fontsize=title_fontsize)

    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax3.plot(L3_forced_Array[idx], label='L3', color=colors[2])
    ax3.legend(loc='upper right', fontsize=legend_fontsize)
    ax3.set_xlabel('Epochs', fontsize=label_fontsize)
    ax3.set_ylabel('Loss', fontsize=label_fontsize)
    ax3.set_title('L3', fontsize=title_fontsize)

    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax4.plot(L4_forced_Array[idx], label='L4', color=colors[3])
    ax4.legend(loc='upper right', fontsize=legend_fontsize)
    ax4.set_xlabel('Epochs', fontsize=label_fontsize)
    ax4.set_ylabel('Loss', fontsize=label_fontsize)
    ax4.set_title('L4', fontsize=title_fontsize)

    ax5 = plt.subplot2grid((4, 2), (3, 0))
    ax5.plot(L5_forced_Array[idx], label='L5', color=colors[4])
    ax5.legend(loc='upper right', fontsize=legend_fontsize)
    ax5.set_xlabel('Epochs', fontsize=label_fontsize)
    ax5.set_ylabel('Loss', fontsize=label_fontsize)
    ax5.set_title('L5', fontsize=title_fontsize)

    ax6 = plt.subplot2grid((4, 2), (3, 1))
    ax6.plot(L6_forced_Array[idx], label='L6', color=colors[5])
    ax6.legend(loc='upper right', fontsize=legend_fontsize)
    ax6.set_xlabel('Epochs', fontsize=label_fontsize)
    ax6.set_ylabel('Loss', fontsize=label_fontsize)
    ax6.set_title('L6', fontsize=title_fontsize)

    plt.tight_layout()
    plt.show()

def plot_debug(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T):
    # Set font sizes
    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # Use the validation tensor as xuk
      # Use the validation tensor as xuk
    xuk = val_tensor

    # Get the debug outputs (assumes debug_* functions are defined)
    actual_L1, predicted_L1 = debug_L1(xuk, Num_meas, model)
    actual_L2, predicted_L2 = debug_L2(xuk, Num_meas, model)

    actual_L3, predicted_L3 = debug_L3(xuk, Num_meas, model)
    actual_L4, predicted_L4 = debug_L4(xuk, Num_meas, model)
    actual_L5, predicted_L5 = debug_L5(xuk, Num_meas, S_p, model)
    actual_L6, predicted_L6 = debug_L6(xuk, Num_meas, Num_x_Obsv, T, model)

    # Select three random sample indices
    sample_indices = random.sample(range(xuk.shape[0]), 3)

    # ---------------------------
    # ENCODER/DECODER PLOT (L1)
    # ---------------------------

    num_vars = actual_L1.shape[2]

    fig, axs = plt.subplots(num_vars, len(sample_indices), figsize=(6 * len(sample_indices), 4 * num_vars), sharex=True)

    if num_vars == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L1[idx]  # shape: (time_steps, num_vars)
        actual_traj = actual_L1[idx]          # shape: (time_steps, num_vars)
        time_steps = range(actual_traj.shape[0])

        for var in range(num_vars):
            ax = axs[var, i]
            ax.plot(time_steps, actual_traj[:, var].cpu().numpy(), 'o-', label=f'True $x_{{{var+1}}}$')
            ax.plot(time_steps, predicted_traj[:, var].detach().cpu().numpy(), 'x--', label=f'Predicted $x_{{{var+1}}}$')
            ax.set_title(f"Validation L1 (gx), Sample {idx} (x{var+1})", fontsize=title_fontsize)
            ax.set_xlabel("Time step", fontsize=label_fontsize)
            ax.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # GU VALIDATION PLOT (L2)
    # ---------------------------
    num_vars = actual_L2.shape[2]

    fig, axs = plt.subplots(num_vars, len(sample_indices), figsize=(6 * len(sample_indices), 4 * num_vars), sharex=True)

    if num_vars == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L2[idx]  # shape: (time_steps, num_vars)
        actual_traj = actual_L2[idx]          # shape: (time_steps, num_vars)
        time_steps = range(actual_traj.shape[0])

        for var in range(num_vars):
            ax = axs[var, i]
            ax.plot(time_steps, actual_traj[:, var].cpu().numpy(), 'o-', label=f'True $x_{{{var+1}}}$')
            ax.plot(time_steps, predicted_traj[:, var].detach().cpu().numpy(), 'x--', label=f'Predicted $x_{{{var+1}}}$')
            ax.set_title(f"Validation L2 (gu), Sample {idx} (x{var+1})", fontsize=title_fontsize)
            ax.set_xlabel("Time step", fontsize=label_fontsize)
            ax.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()
    # Select three random sample indices
    sample_indices = random.sample(range(xuk.shape[0]), 3)

    # ---------------------------
    # L3 VALIDATION PLOT (Dynamic)
    # ---------------------------
    zoom_start, zoom_end = 100, 200
    num_vars = actual_L3.shape[1]
    fig, axs = plt.subplots(num_vars, 2, figsize=(18, 4 * num_vars))

    if num_vars == 1:
        axs = axs.reshape(1, -1)

    for var in range(num_vars):
        # Plot for whole data
        ax_whole = axs[var, 0]
        ax_whole.plot(actual_L3[:, var].cpu().numpy(), label=f'True x{var+1}')
        ax_whole.plot(predicted_L3[:, var].detach().cpu().numpy(), label=f'Predicted x{var+1}')
        ax_whole.set_title(f"L3 validation x{var+1} (Whole Data)", fontsize=title_fontsize)
        ax_whole.set_xlabel("Time step", fontsize=label_fontsize)
        ax_whole.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
        ax_whole.legend(fontsize=legend_fontsize)

        # Plot for zoomed-in data
        ax_zoom = axs[var, 1]
        ax_zoom.plot(actual_L3[zoom_start:zoom_end, var].cpu().numpy(), label=f'True x{var+1}')
        ax_zoom.plot(predicted_L3[zoom_start:zoom_end, var].detach().cpu().numpy(), label=f'Predicted x{var+1}')
        ax_zoom.set_title(f"L3 validation x{var+1} (Zoom In)", fontsize=title_fontsize)
        ax_zoom.set_xlabel("Time step", fontsize=label_fontsize)
        ax_zoom.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
        ax_zoom.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()


    # ---------------------------
    # L4 VALIDATION PLOT (Dynamic)
    # ---------------------------

    zoom_start, zoom_end = 100, 200
    num_vars = actual_L4.shape[1]
    fig, axs = plt.subplots(num_vars, 2, figsize=(18, 4 * num_vars))

    if num_vars == 1:
        axs = axs.reshape(1, -1)

    for var in range(num_vars):
        # Whole data plot
        ax_whole = axs[var, 0]
        ax_whole.plot(actual_L4[:, var].detach().cpu().numpy(), label=f'True Y{var+1}')
        ax_whole.plot(predicted_L4[:, var].detach().cpu().numpy(), label=f'Predicted Y{var+1}')
        ax_whole.set_title(f"L4 validation Y{var+1} (Whole Data)", fontsize=title_fontsize)
        ax_whole.set_xlabel("Time step", fontsize=label_fontsize)
        ax_whole.set_ylabel(f"Y{var+1}", fontsize=label_fontsize)
        ax_whole.legend(fontsize=legend_fontsize)

        # Zoom in plot
        ax_zoom = axs[var, 1]
        ax_zoom.plot(actual_L4[zoom_start:zoom_end, var].detach().cpu().numpy(), label=f'True Y{var+1}')
        ax_zoom.plot(predicted_L4[zoom_start:zoom_end, var].detach().cpu().numpy(), label=f'Predicted Y{var+1}')
        ax_zoom.set_title(f"L4 validation Y{var+1} (Zoom In)", fontsize=title_fontsize)
        ax_zoom.set_xlabel("Time step", fontsize=label_fontsize)
        ax_zoom.set_ylabel(f"Y{var+1}", fontsize=label_fontsize)
        ax_zoom.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()


    # ---------------------------
    # L5 VALIDATION PLOT
    # ---------------------------
    num_vars = actual_L5.shape[2]
    print(num_vars)

    fig, axs = plt.subplots(num_vars, len(sample_indices), figsize=(6 * len(sample_indices), 4 * num_vars), sharex=True)

    if num_vars == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L5[idx]  # shape: (time_steps, num_vars)
        actual_traj = actual_L5[idx]          # shape: (time_steps, num_vars)
        time_steps = range(actual_traj.shape[0])

        for var in range(num_vars):
            ax = axs[var, i]
            ax.plot(time_steps, actual_traj[:, var].cpu().numpy(), 'o-', label=f'True')
            ax.plot(time_steps, predicted_traj[:, var].detach().cpu().numpy(), 'x--', label=f'Predicted')
            ax.set_title(f"Validation L5, Sample {idx} (x{var+1})", fontsize=title_fontsize)
            ax.set_xlabel("Time step", fontsize=label_fontsize)
            ax.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # L6 VALIDATION PLOT (Dynamic)
    # ---------------------------
    num_vars = actual_L6.shape[2]

    fig, axs = plt.subplots(num_vars, len(sample_indices),
                            figsize=(6 * len(sample_indices), 4 * num_vars),
                            sharex=True)

    if num_vars == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(sample_indices):
        predicted_traj = predicted_L6[idx]  # shape: (time_steps, num_vars)
        actual_traj = actual_L6[idx]          # shape: (time_steps, num_vars)
        time_steps = range(actual_traj.shape[0])

        for var in range(num_vars):
            ax = axs[var, i]
            # Detach both tensors before converting to numpy
            ax.plot(time_steps, actual_traj[:, var].detach().cpu().numpy(), 'o-', label='True')
            ax.plot(time_steps, predicted_traj[:, var].detach().cpu().numpy(), 'x--', label='Predicted')
            ax.set_title(f"Validation L6, Sample {idx} (x{var+1})", fontsize=title_fontsize)
            ax.set_xlabel("Time step", fontsize=label_fontsize)
            ax.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()


def plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T):

    title_fontsize = 14
    label_fontsize = 12
    legend_fontsize = 10

    # ---------------------------
    # VALIDATION PREDICTIONS (enc_self_feeding)
    # ---------------------------

    sample_indices_val = random.sample(range(val_tensor.shape[0]), 3)
    Val_pred_traj, val_loss = enc_self_feeding(model, val_tensor, Num_meas)
    print(f"Running loss for validation: {val_loss:.3e}")
    val_tensor = val_tensor[:,:,:Num_meas]
    num_vars_val = val_tensor.shape[2]
    fig, axs = plt.subplots(num_vars_val, len(sample_indices_val),
                            figsize=(6 * len(sample_indices_val), 4 * num_vars_val),
                            sharex=True)
    if num_vars_val == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(sample_indices_val):
        predicted_traj = Val_pred_traj[idx]      # Shape: (time_steps, num_vars_val)
        actual_traj = val_tensor[idx]              # Shape: (time_steps, num_vars_val)
        time_steps = range(actual_traj.shape[0])

        for var in range(num_vars_val):
            ax = axs[var, i]
            ax.plot(time_steps, actual_traj[:, var].cpu().numpy(), 'o-', label=f'True x{var+1}')
            ax.plot(time_steps, predicted_traj[:, var].detach().cpu().numpy(), 'x--', label=f'Predicted x{var+1}')
            ax.set_title(f"Validation Sample {idx} (x{var+1})", fontsize=title_fontsize)
            ax.set_xlabel("Time step", fontsize=label_fontsize)
            ax.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # TRAINING PREDICTIONS (enc_self_feeding)
    # ---------------------------

    sample_indices_train = random.sample(range(train_tensor.shape[0]), 3)
    train_pred_traj, train_loss = enc_self_feeding(model, train_tensor, Num_meas)
    print(f"Running loss for training: {train_loss:.3e}")
    train_tensor = train_tensor[:,:,:Num_meas]
    num_vars_train = train_tensor.shape[2]
    fig, axs = plt.subplots(num_vars_train, len(sample_indices_train),
                            figsize=(6 * len(sample_indices_train), 4 * num_vars_train),
                            sharex=True)
    if num_vars_train == 1:
        axs = axs.reshape(1, -1)

    for i, idx in enumerate(sample_indices_train):
        predicted_traj = train_pred_traj[idx]    # Shape: (time_steps, num_vars_train)
        actual_traj = train_tensor[idx]            # Shape: (time_steps, num_vars_train)
        time_steps = range(actual_traj.shape[0])

        for var in range(num_vars_train):
            ax = axs[var, i]
            ax.plot(time_steps, actual_traj[:, var].cpu().numpy(), 'o-', label=f'True x{var+1}')
            ax.plot(time_steps, predicted_traj[:, var].detach().cpu().numpy(), 'x--', label=f'Predicted x{var+1}')
            ax.set_title(f"Train Sample {idx} (x{var+1})", fontsize=title_fontsize)
            ax.set_xlabel("Time step", fontsize=label_fontsize)
            ax.set_ylabel(f"x{var+1}", fontsize=label_fontsize)
            ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.show()
