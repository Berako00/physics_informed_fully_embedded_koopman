import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os


from help_func import self_feeding, enc_self_feeding, set_requires_grad, get_model_path, enc_self_feeding_uf, load_model
from loss_func import total_loss, total_loss_forced, total_loss_unforced
from nn_structure import AUTOENCODER


def trainingfcn(eps, check_epoch, lr, batch_size, S_p, T, dt, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv,
                Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_u_encoder, train_tensor, test_tensor, M, device=None):

  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  pin_memory = True if device.type == "cuda" else False

  train_dataset = TensorDataset(train_tensor)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
  test_dataset = TensorDataset(test_tensor)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

  Models_loss_list = torch.zeros(M)
  c_m = 0

  Model_path = [get_model_path(i) for i in range(M)]
  Running_Losses_Array, L4_Array, L6_Array = [torch.zeros(M, eps) for _ in range(3)]

  hyperparams = {
        'Num_meas': Num_meas,
        'Num_inputs': Num_inputs,
        'Num_x_Obsv': Num_x_Obsv,
        'Num_x_Neurons': Num_x_Neurons,
        'Num_u_Obsv': Num_u_Obsv,
        'Num_u_Neurons': Num_u_Neurons,
        'Num_hidden_x_encoder': Num_hidden_x_encoder,
        'Num_hidden_u_encoder': Num_hidden_u_encoder,
        'dt': dt
  }

  for c_m in range(M):
      model_path_i = Model_path[c_m]
      model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv,
                          Num_x_Neurons, Num_u_Obsv, Num_u_Neurons,
                          Num_hidden_x_encoder,
                          Num_hidden_u_encoder).to(device)

      optimizer = optim.Adam(model.parameters(), lr=lr)

      best_test_loss_checkpoint = float('inf')

      running_loss_list, L4_list, L6_list = [torch.zeros(eps) for _ in range(3)]

      for e in range(eps):
          model.train()
          running_loss, running_L4, running_L6 = [0.0] * 3

          for (batch_x,) in train_loader:
              batch_x = batch_x.to(device, non_blocking=True)
              optimizer.zero_grad()

              [loss, L_4, L_6] = total_loss(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)
              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              running_L4 += L_4.item()
              running_L6 += L_6.item()

              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

          running_loss_list[e] = running_loss
          L4_list[e] = running_L4
          L6_list[e] = running_L6

          # Every x epochs, evaluate on the test set and checkpoint if improved.
          if (e + 1) % check_epoch == 0:
              model.eval()
              test_running_loss = 0.0
              for (batch_x,) in test_loader:
                  batch_x = batch_x.to(device, non_blocking=True)
                  _, loss = enc_self_feeding(model, batch_x, Num_meas)
                  test_running_loss += loss.item()
              print(f'Checkpoint at Epoch {e+1}: Test Running Loss: {test_running_loss:.3e}')

              # If test loss is lower than the one from the previous checkpoint, save the model.
              if test_running_loss < best_test_loss_checkpoint:
                  best_test_loss_checkpoint = test_running_loss
                  checkpoint = {'state_dict': model.state_dict(), **hyperparams}
                  torch.save(checkpoint, model_path_i)
                  print(f'Checkpoint at Epoch {e+1}: New best test loss, model saved.')

      # load_model(model, model_path_i, device)

      Models_loss_list[c_m] = best_test_loss_checkpoint
      Running_Losses_Array[c_m, :] = running_loss_list
      L4_Array[c_m, :] = L4_list
      L6_Array[c_m, :] = L6_list

  # Find the best of the models
  Lowest_loss = Models_loss_list.min().item()

  Lowest_loss_index = int((Models_loss_list == Models_loss_list.min()).nonzero(as_tuple=False)[0].item())
  # print(f"The best model has a running loss of {Lowest_loss:.3e} and is model nr. {Lowest_loss_index}")

  Best_Model = Model_path[Lowest_loss_index]

  return (Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index, Running_Losses_Array, L4_Array, L6_Array)

def trainingfcn_ga(eps, check_epoch, lr, batch_size, S_p, T, dt, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv,
                Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_u_encoder, train_tensor, test_tensor, M, device=None):

  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  pin_memory = True if device.type == "cuda" else False

  train_dataset = TensorDataset(train_tensor)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
  test_dataset = TensorDataset(test_tensor)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

  Models_loss_list = torch.zeros(M)
  c_m = 0

  Model_path = [get_model_path(i) for i in range(M)]
  Running_Losses_Array, L4_Array, L6_Array = [torch.zeros(M, eps) for _ in range(3)]

  hyperparams = {
        'Num_meas': Num_meas,
        'Num_inputs': Num_inputs,
        'Num_x_Obsv': Num_x_Obsv,
        'Num_x_Neurons': Num_x_Neurons,
        'Num_u_Obsv': Num_u_Obsv,
        'Num_u_Neurons': Num_u_Neurons,
        'Num_hidden_x_encoder': Num_hidden_x_encoder,
        'Num_hidden_u_encoder': Num_hidden_u_encoder,
        'dt': dt
  }

  for c_m in range(M):
      model_path_i = Model_path[c_m]
      model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv,
                          Num_x_Neurons, Num_u_Obsv, Num_u_Neurons,
                          Num_hidden_x_encoder,
                          Num_hidden_u_encoder).to(device)

      optimizer = optim.Adam(model.parameters(), lr=lr)

      best_test_loss_checkpoint = float('inf')

      running_loss_list, L4_list, L6_list = [torch.zeros(eps) for _ in range(3)]

      for e in range(eps):
          model.train()
          running_loss, running_L4, running_L6 = [0.0] * 3

          for (batch_x,) in train_loader:
              batch_x = batch_x.to(device, non_blocking=True)
              optimizer.zero_grad()

              [loss, L_4, L_6] = total_loss(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)
              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              running_L4 += L_4.item()
              running_L6 += L_6.item()

              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

          running_loss_list[e] = running_loss
          L4_list[e] = running_L4
          L6_list[e] = running_L6

          # Every x epochs, evaluate on the test set and checkpoint if improved.
          if (e + 1) % check_epoch == 0:
              # print(f"Epoch {e+1}/{eps}, "f"Train Loss: {running_loss:.3e}, "f"Elapsed: {elapsed_str}, ETA: {eta_str}")
              model.eval()
              test_running_loss = 0.0
              for (batch_x,) in test_loader:
                  batch_x = batch_x.to(device, non_blocking=True)
                  _, loss = enc_self_feeding(model, batch_x, Num_meas)
                  test_running_loss += loss.item()
              # print(f'Checkpoint at Epoch {e+1}: Test Running Loss: {test_running_loss:.3e}')

              # If test loss is lower than the one from the previous checkpoint, save the model.
              if test_running_loss < best_test_loss_checkpoint:
                  best_test_loss_checkpoint = test_running_loss
                  # checkpoint = {'state_dict': model.state_dict(), **hyperparams}
                  # torch.save(checkpoint, model_path_i)
                  # print(f'Checkpoint at Epoch {e+1}: New best test loss, model saved.')

      # load_model(model, model_path_i, device)

      Models_loss_list[c_m] = best_test_loss_checkpoint
      Running_Losses_Array[c_m, :] = running_loss_list
      L4_Array[c_m, :] = L4_list
      L6_Array[c_m, :] = L6_list

  # Find the best of the models
  Lowest_loss = Models_loss_list.min().item()

  Lowest_loss_index = int((Models_loss_list == Models_loss_list.min()).nonzero(as_tuple=False)[0].item())
  # print(f"The best model has a running loss of {Lowest_loss:.3e} and is model nr. {Lowest_loss_index}")

  Best_Model = Model_path[Lowest_loss_index]

  return (Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index, Running_Losses_Array, L4_Array, L6_Array)

###

def trainingfcn_mixed(eps,check_epoch, lr, batch_size, S_p, T, alpha, Num_meas, Num_inputs, Num_x_Obsv,
                      Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor_unforced, train_tensor_forced,
                      test_tensor_unforced, test_tensor_forced, M, device=None):
  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
  pin_memory = True if device.type == "cuda" else False

  hyperparams = {
        'Num_meas': Num_meas,
        'Num_inputs': Num_inputs,
        'Num_x_Obsv': Num_x_Obsv,
        'Num_x_Neurons': Num_x_Neurons,
        'Num_u_Obsv': Num_u_Obsv,
        'Num_u_Neurons': Num_u_Neurons,
        'Num_hidden_x_encoder': Num_hidden_x_encoder,
        'Num_hidden_u_encoder': Num_hidden_u_encoder
  }

  train_unforced_dataset = TensorDataset(train_tensor_unforced)
  train_unforced_loader = DataLoader(train_unforced_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

  train_forced_dataset = TensorDataset(train_tensor_forced)
  train_forced_loader = DataLoader(train_forced_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

  test_unforced_dataset = TensorDataset(test_tensor_unforced)
  test_unforced_loader = DataLoader(test_unforced_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

  test_forced_dataset = TensorDataset(test_tensor_forced)
  test_forced_loader = DataLoader(test_forced_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

  Models_loss_list = torch.zeros(M)
  [Running_Losses_Array, Lgu_forced_Array, L4_unforced_Array, L4_forced_Array, L6_unforced_Array, L6_forced_Array] = [torch.zeros(M, eps) for _ in range(6)]

  c_m = 0

  Model_path = [get_model_path(i) for i in range(M)]

  for c_m in range(M):

      model_path_i = Model_path[c_m]
      model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv,
                          Num_x_Neurons, Num_u_Obsv, Num_u_Neurons,
                          Num_hidden_x_encoder,
                          Num_hidden_u_encoder, Num_hidden_u_decoder).to(device)

      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
      best_test_loss_checkpoint = float('inf')

      [running_loss_list, Lgu_forced_list,
       L4_unforced_list, L4_forced_list,
       L6_unforced_list, L6_forced_list] = [torch.zeros(eps) for _ in range(6)]

      #First train the unforced system so do not compute
      set_requires_grad(list(model.u_Encoder_In.parameters()) +
                        list(model.u_encoder_hidden.parameters()) +
                        list(model.u_Encoder_out.parameters()) +
                        list(model.u_Koopman.parameters()) +
                        list(model.u_Decoder_In.parameters()) +
                        list(model.u_decoder_hidden.parameters()) +
                        list(model.u_Decoder_out.parameters()), requires_grad=False)

      for e in range(eps):
          model.train()
          running_loss, running_Lgx, running_Lgu, running_L3, running_L4, running_L5, running_L6 = [0.0] * 7

          for (batch_x,) in train_unforced_loader:
              batch_x = batch_x.to(device, non_blocking=True)
              optimizer.zero_grad()
              [loss, L_4, L_6] = total_loss_unforced(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)

              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              running_L4 += L_4.item()
              running_L6 += L_6.item()

              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

          running_loss_list[e] = running_loss # This one may be deleted
          L4_unforced_list[e] = running_L4
          L6_unforced_list[e] = running_L6

          print(f'Input: 0, Model: {c_m}, Epoch {e+1}, Running loss: {running_loss:.3e}')

          if (e + 1) % check_epoch == 0:
              model.eval()
              test_running_loss = 0.0
              for (batch_x,) in test_unforced_loader:
                  batch_x = batch_x.to(device, non_blocking=True)
                  _, loss = enc_self_feeding_uf(model, batch_x, Num_meas)
                  test_running_loss += loss.item()
              print(f'Checkpoint at Epoch {e+1}: Test Running Loss: {test_running_loss:.3e}')

              # If test loss is lower than the one from the previous checkpoint, save the model.
              if test_running_loss < best_test_loss_checkpoint:
                  best_test_loss_checkpoint = test_running_loss
                  checkpoint = {'state_dict': model.state_dict(), **hyperparams}
                  torch.save(checkpoint, model_path_i)
                  print(f'Checkpoint at Epoch {e+1}: New best test loss, model saved.')

      load_model(model, model_path_i, device)

      set_requires_grad(model.parameters(), requires_grad=False) # Set all parames to not train
      #Enable training of forced system
      set_requires_grad(list(model.u_Encoder_In.parameters()) +
                        list(model.u_encoder_hidden.parameters()) +
                        list(model.u_Encoder_out.parameters()) +
                        list(model.u_Koopman.parameters()) +
                        list(model.u_Decoder_In.parameters()) +
                        list(model.u_decoder_hidden.parameters()) +
                        list(model.u_Decoder_out.parameters()), requires_grad=True)


      optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

      best_test_loss_checkpoint = float('inf')

      for e in range(eps):
          model.train()
          running_loss, running_Lgu, running_L4, running_L6 = [0.0] * 4

          for (batch_x,) in train_forced_loader:
              batch_x = batch_x.to(device, non_blocking=True)
              optimizer.zero_grad()
              [loss, L_gu, L_4, L_6] = total_loss_forced(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)

              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              running_Lgu += L_gu.item()
              running_L4 += L_4.item()
              running_L6 += L_6.item()

              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

          running_loss_list[e] = running_loss
          Lgu_forced_list[e] = running_Lgu
          L4_forced_list[e] = running_L4
          L6_forced_list[e] = running_L6

          print(f'Variable Input, Model: {c_m}, Epoch {e+1}, Running loss: {running_loss:.3e}')

          if (e + 1) % check_epoch == 0:
              model.eval()
              test_running_loss = 0.0
              for (batch_x,) in test_forced_loader:
                  batch_x = batch_x.to(device, non_blocking=True)
                  _, loss = enc_self_feeding(model, batch_x, Num_meas)
                  test_running_loss += loss.item()
              print(f'Checkpoint at Epoch {e+1}: Test Running Loss: {test_running_loss:.3e}')

              # If test loss is lower than the one from the previous checkpoint, save the model.
              if test_running_loss < best_test_loss_checkpoint:
                  best_test_loss_checkpoint = test_running_loss
                  checkpoint = {'state_dict': model.state_dict(), **hyperparams}
                  torch.save(checkpoint, model_path_i)
                  print(f'Checkpoint at Epoch {e+1}: New best test loss, model saved.')

      load_model(model, model_path_i, device)

      Models_loss_list[c_m] = best_test_loss_checkpoint
      Running_Losses_Array[c_m, :] = running_loss_list
      Lgu_forced_Array[c_m, :] = Lgu_forced_list

      L4_unforced_Array[c_m, :] = L4_unforced_list
      L6_unforced_Array[c_m, :] = L6_unforced_list

      L4_forced_Array[c_m, :] = L4_forced_list
      L6_forced_Array[c_m, :] = L6_forced_list

  # Find the best of the models
  Lowest_loss = Models_loss_list.min().item()

  Lowest_loss_index = int((Models_loss_list == Models_loss_list.min()).nonzero(as_tuple=False)[0].item())
  print(f"The best model has a running loss of {Lowest_loss:.3e} and is model nr. {Lowest_loss_index}")

  Best_Model = Model_path[Lowest_loss_index]

  return (Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index,
          Running_Losses_Array, Lgu_forced_Array,
          L4_unforced_Array, L6_unforced_Array,
          L4_forced_Array, L6_forced_Array)
