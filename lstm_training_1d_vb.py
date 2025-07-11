import warnings
import argparse
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from generate_data import collect_multiparam_snapshots_test, collect_multiparam_snapshots_train
import torch.nn.functional as F
# Set seeds
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description="Run Autoencoder Training")
parser.add_argument("--method", type=str, default="tunable", help="Specify the method to use (e.g., 'ae', 'vae', etc.)")
parser.add_argument("--latent", type=int, default=2, help="Specify the method to use (e.g., 'ae', 'vae', etc.)")
args = parser.parse_args()

grid = 128
Rnum = 1000
num_time_steps = 100
x = np.linspace(0.0, 1.0, num=128)
dx = 1.0 / np.shape(x)[0]
tsteps = np.linspace(0.0, 2.0, num=num_time_steps)
dt = 2.0 / np.shape(tsteps)[0]
time_window = 10  # The window size of the LSTM
mode = 'test'
encoded_space_dim = args.latent
lrate = 0.001
device = 'cuda'
method = args.method

snapshots_train, rnum_vals_train = collect_multiparam_snapshots_train(x, tsteps)  # So that dim=0 corresponds to number of snapshots
snapshots_test, rnum_vals_test = collect_multiparam_snapshots_test(x, tsteps)  # So that dim=0 corresponds to number of snapshots

snapshots_train = np.expand_dims(snapshots_train, -1)
snapshots_test = np.expand_dims(snapshots_test, -1)



def get_train_svd(X_train):
    X_train = X_train[:,:,0]
    u, s, vh = np.linalg.svd(X_train, full_matrices=False)
    vhr = vh[:encoded_space_dim, :]
    vhr = torch.from_numpy(vhr).float().to(device)

    return vhr


vhr = get_train_svd(snapshots_train)

def coeff_determination(y_true, y_pred):
    """
    Compute the coefficient of determination (R² score) in PyTorch.

    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: R² score.
    """
    SS_res = torch.sum((y_true - y_pred) ** 2)
    SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - SS_res / (SS_tot + 1e-7)

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, dropout_prob=0.4):
        super(Encoder, self).__init__()
        
        self.fc = nn.Sequential(
           torch.tanh(nn.Linear(128 , 2*encoded_space_dim)),
           nn.Linear(2*encoded_space_dim, encoded_space_dim),
           
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dropout_prob=0.4):
        super(Decoder, self).__init__()
        
        self.fc = nn.Sequential(
           torch.tanh(nn.Linear(encoded_space_dim, 2*encoded_space_dim)),
           nn.Linear(2*encoded_space_dim, 128),
           
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x


# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        if method=='tunable':
            self.a = nn.Parameter(torch.zeros(encoded_space_dim))
            self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x):

        if method=='ae':
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        elif method=='naive':
            encoded = self.encoder(x) + (x.squeeze(1) @ vhr.T)
            decoded = self.decoder(encoded) + ((encoded.squeeze(1) @ vhr).unsqueeze(1))
        elif method == 'tunable':
            encoded = self.a * self.encoder(x) + (1.0 - self.a) * (x.squeeze(1) @ vhr.T)
            decoded = self.b * self.decoder(encoded) + (1.0 - self.b) * ((encoded.squeeze(1) @ vhr).unsqueeze(1))
        elif method=='pod':
            encoded = (x.squeeze(1) @ vhr.T)
            decoded = ((encoded.squeeze(1) @ vhr).unsqueeze(1))

        return decoded

model = Autoencoder()
model = model.to(device)

# Loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Training Loop
num_epochs = 500
batch_size = 32

# Assume `snapshots_train` is a PyTorch tensor of shape (N, 1, 128)
snapshots_train = torch.tensor(snapshots_train, dtype=torch.float32)
snapshots_train = snapshots_train.permute((0,2,1)).to(device)
train_loader = DataLoader(snapshots_train, batch_size=batch_size, shuffle=True)


scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-4, max_lr = 1e-1)

if mode=='train' and method!='pod':

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # Track total loss for the epoch

        for batch in train_loader:  # Iterate over batches
            batch = batch.to(device)  # Move batch to the same device as the model

            optimizer.zero_grad()

            # Forward pass
            output = model(batch)

            # Compute loss
            loss = criterion(output, batch)
            epoch_loss += loss.item()  # Accumulate batch loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Step the learning rate scheduler
        scheduler.step(epoch_loss/len(train_loader))

        # Compute R² score on the entire dataset (optional)


        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                all_outputs = model(snapshots_train.to(device))
                r2 = coeff_determination(snapshots_train.to(device), all_outputs)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, R²: {r2.item():.4f}')


    torch.save(model.state_dict(), f'ae_params_{method}.pt')
    with torch.no_grad():
        sim_num = 2
        recoded_1 = model.forward(
            snapshots_train[(sim_num + 1) * (num_time_steps - 1):(sim_num + 1) * (num_time_steps - 1) + 1, :, :])
        sim_num = 12
        recoded_2 = model.forward(
            snapshots_train[(sim_num + 1) * (num_time_steps - 1):(sim_num + 1) * (num_time_steps - 1) + 1, :, :])

else:
    if method!='pod':
        model.load_state_dict(torch.load(f'./ae_params_{method}.pt'))
    with torch.no_grad():
        sim_num = 2
        recoded_1 = model.forward(
            snapshots_train[(sim_num + 1) * (num_time_steps - 1):(sim_num + 1) * (num_time_steps - 1) + 1, :, :])
        sim_num = 12
        recoded_2 = model.forward(
            snapshots_train[(sim_num + 1) * (num_time_steps - 1):(sim_num + 1) * (num_time_steps - 1) + 1, :, :])



sim_num = 2
fig, ax = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
ax[0].plot(snapshots_train[(sim_num + 1) * (num_time_steps - 1), 0, :].cpu().numpy(), 'b', label='Truth', linewidth=3)
ax[0].plot(recoded_1[0, 0, :].cpu().numpy(), 'ro', label='Reconstructed', markersize=3)

sim_num = 12
ax[1].plot(snapshots_train[(sim_num + 1) * (num_time_steps - 1), 0, :].cpu().numpy(), 'b', label='Truth', linewidth=3)
ax[1].plot(recoded_2[0, 0, :].cpu().numpy(), 'ro', label='Reconstructed', markersize=3)

ax[0].legend()

ax[0].set_ylim((0, 0.5))
ax[1].set_ylim((0, 0.5))

plt.show()

snapshots_test = torch.tensor(snapshots_test, dtype=torch.float32).to(device).permute((0,2,1))
before_lstm = model.forward(snapshots_test)
np.save(f'./test_data_before_lstm_{method}.npy', snapshots_test.cpu().numpy().reshape((13, 100, 128))[:,10:,:].reshape((13*90, 1, 128)))
np.save(f'./test_prediction_before_lstm_{method}.npy', before_lstm.detach().cpu().numpy().reshape((13, 100, 128))[:,10:,:].reshape((13*90, 1, 128)))
with torch.no_grad():
    model.eval()
    if method=='ae':
        encoded = model.encoder(snapshots_train)
        encoded_test = model.encoder(snapshots_test)
    elif method == 'naive':
        encoded = model.encoder(snapshots_train) + (snapshots_train.squeeze(1) @ vhr.T)
        encoded_test = model.encoder(snapshots_test) + (snapshots_test.squeeze(1) @ vhr.T)
    elif method == 'tunable':
        encoded = model.a * model.encoder(snapshots_train) + (1.0 - model.a) * (snapshots_train.squeeze(1) @ vhr.T)
        encoded_test = model.a * model.encoder(snapshots_test) + (1.0 - model.a) * (snapshots_test.squeeze(1) @ vhr.T)
    elif method=='pod':
        encoded = (snapshots_train.squeeze(1) @ vhr.T)
        encoded_test = (snapshots_test.squeeze(1) @ vhr.T)

encoded = encoded.cpu().numpy().reshape((len(rnum_vals_train), num_time_steps, encoded_space_dim))
encoded_test = encoded_test.cpu().numpy().reshape((len(rnum_vals_test), num_time_steps, encoded_space_dim))

encoded_f = np.copy(encoded)
encoded_test_f = np.copy(encoded_test)

from scipy.ndimage.filters import gaussian_filter1d

for rnum in range(len(rnum_vals_train)):
    encoded_f[rnum, :, 0] = gaussian_filter1d(encoded[rnum, :, 0], sigma=3)
    encoded_f[rnum, :, 1] = gaussian_filter1d(encoded[rnum, :, 1], sigma=3)

for rnum in range(len(rnum_vals_test)):
    encoded_test_f[rnum, :, 0] = gaussian_filter1d(encoded_test[rnum, :, 0], sigma=3)
    encoded_test_f[rnum, :, 1] = gaussian_filter1d(encoded_test[rnum, :, 1], sigma=3)

num_train_snapshots = np.shape(rnum_vals_train)[0]

rnum_tracker = np.zeros(shape=(num_train_snapshots, num_time_steps, 1))
for i in range(np.shape(rnum_vals_train)[0]):
    rnum_tracker[i, :, 0] = rnum_vals_train[i]

lstm_training_data = np.concatenate((encoded, rnum_tracker), axis=-1)

total_size = np.shape(lstm_training_data)[0] * np.shape(lstm_training_data)[1]
total_size_sim = np.shape(lstm_training_data)[1]

from sklearn.preprocessing import MinMaxScaler

scale_lstm = False
if scale_lstm:
    scaler = MinMaxScaler()
    lstm_training_data_scaled = scaler.fit_transform(lstm_training_data[0, :, :])
else:
    lstm_training_data_scaled = lstm_training_data[0, :, :]

# Shape the inputs and outputs
input_seq = np.zeros(shape=(total_size - num_train_snapshots * time_window, time_window, encoded_space_dim+1))
output_seq = np.zeros(shape=(total_size - num_train_snapshots * time_window, encoded_space_dim))

# Setting up inputs
sample = 0
for snum in range(num_train_snapshots):
    for t in range(time_window, total_size_sim):
        input_seq[sample, :, :] = lstm_training_data[snum, t - time_window:t, :]
        output_seq[sample, :] = lstm_training_data[snum, t, 0:encoded_space_dim]
        sample = sample + 1

# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Output and hidden states
        if len(lstm_out.shape) == 3:  # Use only the last time step if sequences are required
            x = lstm_out[:, -1, :]  # Take the last time step
        x = self.fc(x)
        return x


input_dim = encoded_space_dim + 1  # Number of features in the input sequence
hidden_dim = 40
output_dim = encoded_space_dim
num_layers = 2
num_epochs = 400
batch_size = 32
learning_rate = 0.001

# Model, Loss, and Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

# input_seq = np.load('./input_seq_from_tf.npy')
# output_seq = np.load('./output_seq_from_tf.npy')
# Dataset and DataLoader
input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)  # Input sequences
output_seq = torch.tensor(output_seq, dtype=torch.float32).to(device)  # Corresponding outputs

dataset = TensorDataset(input_seq, output_seq)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 1e-4, max_lr = 1e-1)

# Training Loop
if mode == 'train':
    for epoch in range(num_epochs):
        lstm_model.train()
        epoch_loss = 0.0
        for batch_inputs, batch_outputs in train_loader:
            optimizer.zero_grad()

            # Forward pass
            predictions = lstm_model(batch_inputs)
            loss = criterion(predictions, batch_outputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Adjust learning rate
        scheduler.step(epoch_loss / len(train_loader))

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}')

    torch.save(lstm_model.state_dict(), f'./lstm_model_{method}.pt')

else:
    lstm_model.load_state_dict(torch.load(f'./lstm_model_{method}.pt'))
    lstm_model.eval()

num_test_snapshots = np.shape(rnum_vals_test)[0]

rnum_tracker = np.zeros(shape=(num_test_snapshots, num_time_steps, 1))
for i in range(np.shape(rnum_vals_test)[0]):
    rnum_tracker[i, :, 0] = rnum_vals_test[i]

lstm_testing_data = np.concatenate((encoded_test, rnum_tracker), axis=-1)

print(lstm_testing_data.shape)
total_size = np.shape(lstm_testing_data)[0] * np.shape(lstm_testing_data)[1]
total_size_sim = np.shape(lstm_testing_data)[1]

scale_lstm = False
for i in range(np.shape(rnum_vals_test)[0]):
    if scale_lstm:
        lstm_testing_data[i, :, :] = scaler.transform(lstm_testing_data[i, :, :])

# Shape the inputs and outputs
input_seq = torch.zeros((1, time_window, 3)).to(device)
output_seq = torch.zeros((total_size - num_test_snapshots * time_window, encoded_space_dim)).to(device)
output_seq_pred = torch.zeros((total_size - num_test_snapshots * time_window, encoded_space_dim)).to(device)

sample = 0
# Loop over testing snapshots
with torch.no_grad():  # Disable gradient computation for inference
    for snum in range(num_test_snapshots):
        # Initialize the input sequence for the snapshot
        input_seq[0, :, :] = torch.tensor(lstm_testing_data[snum, 0:time_window, :], dtype=torch.float32).to(device)

        for t in range(time_window, total_size_sim):
            # Make a prediction using the model
            pred = lstm_model(input_seq)  # Model prediction for the current input sequence

            # Store the predicted and true output
            output_seq_pred[sample, :] = pred[0, :]
            output_seq[sample, :] = torch.tensor(lstm_testing_data[snum, t, 0:encoded_space_dim], dtype=torch.float32).to(device)

            # Shift the input sequence and append the new prediction
            input_seq[0, 0:time_window - 1, :] = input_seq[0, 1:, :].clone()  # Shift left
            input_seq[0, time_window - 1, :encoded_space_dim] = output_seq_pred[sample, :].clone()  # Update the last time step with the prediction
            sample += 1

#
# for sim_num in range(1, num_test_snapshots):
#     fig, ax = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
#
#     ax[0].plot(
#         output_seq_pred.cpu().numpy()[sim_num * (num_time_steps - time_window):(sim_num + 1) * (num_time_steps - time_window), 0],
#         label='True', linewidth=3)
#     ax[0].plot(lstm_testing_data[sim_num, time_window:, 0], 'r--', label='Predicted', linewidth=3)
#
#     ax[1].plot(
#         output_seq_pred.cpu().numpy()[sim_num * (num_time_steps - time_window):(sim_num + 1) * (num_time_steps - time_window), 1],
#         label='True', linewidth=3)
#     ax[1].plot(lstm_testing_data[sim_num, time_window:, 1], 'r--', label='Predicted', linewidth=3)
#
#     ax[0].legend()
#     # ax[0].set_ylim((-1, 1))
#     # ax[1].set_ylim((-1, 1))
#     plt.show()
# %%
# fig, ax = plt.subplots(figsize=(14, 6), nrows=1, ncols=2)
#
# sim_num = 1
# ax[0].plot(output_seq_pred.cpu().numpy()[sim_num * (num_time_steps - time_window):(sim_num + 1) * (num_time_steps - time_window), 0],
#            label='True $Re=250$', linewidth=3)
# ax[0].plot(lstm_testing_data[sim_num, time_window:, 0], 'r--', label=r'Predicted $Re=250$', linewidth=3)
#
# ax[1].plot(output_seq_pred.cpu().numpy()[sim_num * (num_time_steps - time_window):(sim_num + 1) * (num_time_steps - time_window), 1],
#            label='True $Re=250$', linewidth=3)
# ax[1].plot(lstm_testing_data[sim_num, time_window:, 1], 'r--', label=r'Predicted $Re=250$', linewidth=3)
#
# sim_num = num_test_snapshots - 1
# ax[0].plot(output_seq_pred.cpu().numpy()[sim_num * (num_time_steps - time_window):(sim_num + 1) * (num_time_steps - time_window), 0],
#            label='True $Re=2300$', linewidth=3, color='green')
# ax[0].plot(lstm_testing_data[sim_num, time_window:, 0], 'b--', label=r'Predicted $Re=2300$', linewidth=3)
#
# ax[1].plot(output_seq_pred.cpu().numpy()[sim_num * (num_time_steps - time_window):(sim_num + 1) * (num_time_steps - time_window), 1],
#            label='True $Re=2300$', linewidth=3, color='green')
# ax[1].plot(lstm_testing_data[sim_num, time_window:, 1], 'b--', label=r'Predicted $Re=2300$', linewidth=3)
#
# ax[0].legend()
# # ax[0].set_ylim((-1, 1))
# # ax[1].set_ylim((-1, 1))
#
# ax[0].set_xlabel('Temporal snapshot')
# ax[0].set_ylabel('Magnitude')
#
# ax[1].set_xlabel('Temporal snapshot')
# ax[1].set_ylabel('Magnitude')
#
# plt.tight_layout()
# plt.show()

output_seq_pred = output_seq_pred.unsqueeze(1)
with torch.no_grad():
    if method=='ae':
        decoded = model.decoder(output_seq_pred)
    elif method=='naive':
        decoded = model.decoder(output_seq_pred) + ((output_seq_pred.squeeze(1) @ vhr).unsqueeze(1))
    elif method=='tunable':
        decoded = model.b * model.decoder(output_seq_pred) + (1.0 - model.b) * ((output_seq_pred.squeeze(1) @ vhr).unsqueeze(1))
    elif method=='pod':
        decoded = ((output_seq_pred.squeeze(1) @ vhr).unsqueeze(1))

decoded = decoded.cpu().numpy().reshape(len(rnum_vals_test), num_time_steps - time_window, 128)
snapshot_test_true = snapshots_test[:, :, 0:].reshape(len(rnum_vals_test), num_time_steps, 128)
np.save(f'test_data_after_lstm_{method}.npy', snapshot_test_true.cpu().numpy()[:,10:,:].reshape(len(rnum_vals_test)*90, 1, 128))
np.save(f'test_prediction_after_lstm_{method}.npy', decoded.reshape(len(rnum_vals_test)*90, 1, 128))

snapshot_test_true = np.load(f'Testing_data_{method}.npy')
snapshot_test_pred = np.load(f'Testing_data_prediction_{method}.npy')

x = np.arange(128) / 128
snapshot_test_pred = np.copy(decoded)
num_sims = np.shape(snapshot_test_true)[0]
animate = False

test_num = -1
snap = 10
if animate:
    for i in range(time_window, num_time_steps, int(num_time_steps / 9)):
        plt.figure()
        plt.plot(x, decoded[test_num, i - time_window, :], 'ro', label='Reconstructed', markersize=3)
        plt.plot(x, snapshot_test_true[test_num, i, :], label='True', linewidth=3)
        plt.ylim((0.0, 0.5))
        if i == time_window:
            plt.legend(loc='upper right')

        plt.xlabel('x')
        plt.ylabel('u')
        plt.savefig(f'{method}/MP_Reconstruction_' + str(snap) + '.png')
        snap = snap + 10
        plt.show()
# %%
# for sim in range(1, num_sims):
#     plt.figure(figsize=(7, 6))
#     plt.plot(x, decoded[sim, -1, :], 'ro', label='Prediction', markersize=3)
#     plt.plot(x, snapshot_test_true[sim, -1, :], label='True', linewidth=3)
#
#     plt.ylim((0.0, 0.5))
#     if sim == 1:
#         plt.legend()
#
#     plt.xlabel('x')
#     plt.ylabel('u')
#     plt.savefig(f'{method}/MP_Reconstruction_tf_' + str(sim) + '.png')
#     plt.show()
# %%

print(np.mean((snapshot_test_true[:, 10:, :] - snapshot_test_pred[:, :, :]) ** 2))
plt.plot(snapshot_test_true[-1,-1], '-b')
plt.plot(snapshot_test_pred[-1,-1], '.r')
plt.show()
np.save(f'./test_error_{method}.npy', np.mean((snapshot_test_true[:, 10:, :] - snapshot_test_pred[:, :, :]) ** 2))
