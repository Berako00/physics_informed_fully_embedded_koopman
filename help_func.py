import torch
import torch.nn.functional as F
import os

from loss_func import custom_loss

def get_model_path(i):
    windows_dir1 = r"C:\Users\jokin\Desktop\Uni\Aalborg\Master\Masters_Thesis"
    windows_dir2 = r"C:\Users\Labuser\Desktop\WAPNN"
    linux_dir   = "/home/trarity/master/koopman_control/data"
    linux_dir2   = "/home/trarity/state_koopman_operator"
    colab_dir1  = "/content/drive/My Drive/Colab Notebooks"
    colab_dir2  = "/content/drive/MyDrive/Colab Notebooks"  
    ucloud = "/work/state_koopman_operator"
    
    path1 = os.path.join(linux_dir2, f"Autoencoder_model_params{i}.pth")
    path2 = os.path.join(windows_dir1, f"Autoencoder_model_params{i}.pth")
    path3 = os.path.join(colab_dir1, f"Autoencoder_model_params{i}.pth")
    path4 = os.path.join(colab_dir2, f"Autoencoder_model_params{i}.pth")
    path5 = os.path.join(windows_dir2, f"Autoencoder_model_params{i}.pth")
    path6 = os.path.join(ucloud, f"Autoencoder_model_params{i}.pth")

    if os.path.exists(path1):
        chosen_path = path1
    elif os.path.exists(path2):
        chosen_path = path2
    elif os.path.exists(path3):
        chosen_path = path3
    elif os.path.exists(path4):
        chosen_path = path4  
    elif os.path.exists(path5):
        chosen_path = path5     
    else:
        chosen_path = path6
    # print("Using model path:", chosen_path)
    return chosen_path


def set_requires_grad(layers, requires_grad):
    for param in layers:
        param.requires_grad = requires_grad

def self_feeding(model, xuk, Num_meas):
    initial_input = xuk[:, 0, :]
    num_steps = int(len(xuk[0, :, 0]))
    inputs = xuk[:,:,Num_meas:]

    predictions = []
    predictions.append(initial_input)

    for step in range(num_steps - 1):
        x_pred = model(initial_input)
        x_pred = torch.cat((x_pred, inputs[:, step, :]), dim=1)
        predictions.append(x_pred.detach())
        initial_input = x_pred

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk)

    return predictions, loss


def enc_self_feeding(model, xuk, Num_meas):
    x_k = xuk[:, 0, :Num_meas]
    u = xuk[:, :, Num_meas:]

    num_steps = int(len(xuk[0, :, 0]))
    predictions = []
    predictions.append(x_k)

    y_k = model.x_Encoder(x_k)
    for m in range(0, num_steps-1):

        v = model.u_Encoder(torch.cat((x_k, u[:, m, :]), dim=1))
        y_k = model.x_Koopman_op(y_k) + model.u_Koopman_op(v)
        x_k = y_k[:, :Num_meas]
        predictions.append(x_k)

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk[:, :, :Num_meas])

    return predictions, loss

def enc_self_feeding_uf(model, xuk, Num_meas):
    x_k = xuk[:, 0, :Num_meas]

    num_steps = int(len(xuk[0, :, 0]))
    predictions = []
    predictions.append(x_k)

    y_k = model.x_Encoder(x_k)
    for m in range(0, num_steps-1):

        y_k = model.x_Koopman_op(y_k)
        x_k = y_k[:, :Num_meas]
        predictions.append(x_k)

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk[:, :, :Num_meas])

    return predictions, loss
    
def load_model(model, path, device = None):
      checkpoint = torch.load(path, map_location=device)
      if 'state_dict' in checkpoint:
          state_dict = checkpoint['state_dict']
      else:
          state_dict = checkpoint
      model.load_state_dict(state_dict)
      model.eval()

def load_parameters(path, device = None):
      checkpoint = torch.load(path, map_location=device)
      Num_meas = checkpoint['Num_meas']
      Num_inputs = checkpoint['Num_inputs']
      Num_x_Obsv = checkpoint['Num_x_Obsv']
      Num_u_Obsv = checkpoint['Num_u_Obsv']
      Num_x_Neurons = checkpoint['Num_x_Neurons']
      Num_u_Neurons = checkpoint['Num_u_Neurons']
      Num_hidden_x_encoder = checkpoint['Num_hidden_x_encoder']
      Num_hidden_u_encoder = checkpoint['Num_hidden_u_encoder']

      model = AUTOENCODER(Num_meas,Num_inputs,Num_x_Obsv,Num_x_Neurons,
                          Num_u_Obsv,Num_u_Neurons,Num_hidden_x_encoder,
                          Num_hidden_u_encoder)

      if 'state_dict' in checkpoint:
          state_dict = checkpoint['state_dict']
      else:
          state_dict = checkpoint
      model.load_state_dict(state_dict)
      model.eval()

      return model, checkpoint
