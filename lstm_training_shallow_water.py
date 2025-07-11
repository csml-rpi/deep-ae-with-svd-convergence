import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='ae', type=str)
args = parser.parse_args()


# Load data
mode = 'test'
data = np.load('../Equation_Based/snapshot_matrix_pod.npy').T
method = args.method
num_latent = 6
grid = 64
# Scale the training data to zero mean and unit standard deviation
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = data.T
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Make sure the data is in the right format (with channels as the last dimension)
swe_train_data = np.zeros((data.shape[1], 64, 64, 3), dtype=np.float32)
for i in range(data.shape[1]):
    swe_train_data[i, :, :, 0] = data[:64 * 64, i].reshape(64, 64)
    swe_train_data[i, :, :, 1] = data[64 * 64:2 * 64 * 64, i].reshape(64, 64)
    swe_train_data[i, :, :, 2] = data[2 * 64 * 64:, i].reshape(64, 64)

# Convert to PyTorch tensor
swe_train_data = torch.from_numpy(swe_train_data).permute(0, 3, 1, 2)  # Channels first for PyTorch

def get_train_svd(vh):
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # u, s, vh = np.linalg.svd(X_train, full_matrices=False)
    vhr = vh[:num_latent, :]
    vhr = torch.from_numpy(vhr).float().to(device)

    return vhr


vhr = get_train_svd(vh)
# Similar processing for test data
data = np.load('../Equation_Based/snapshot_matrix_test.npy').T
data = scaler.transform(data)
data = data.T

swe_test_data = np.zeros((data.shape[1], 64, 64, 3), dtype=np.float32)
for i in range(data.shape[1]):
    swe_test_data[i, :, :, 0] = data[:64 * 64, i].reshape(64, 64)
    swe_test_data[i, :, :, 1] = data[64 * 64:2 * 64 * 64, i].reshape(64, 64)
    swe_test_data[i, :, :, 2] = data[2 * 64 * 64:, i].reshape(64, 64)

swe_test_data = torch.from_numpy(swe_test_data).permute(0, 3, 1, 2)


# Rescaling function
def unscale(dataset, scaler):
    dataset_phys = np.zeros_like(dataset.numpy())
    for i in range(dataset_phys.shape[0]):
        temp_1 = dataset[i, 0, :, :].numpy().T
        temp_2 = dataset[i, 1, :, :].numpy().T
        temp_3 = dataset[i, 2, :, :].numpy().T
        temp = np.concatenate((temp_1, temp_2, temp_3), axis=0).reshape(1, -1)
        temp = scaler.inverse_transform(temp)
        dataset_phys[i, 0, :, :] = temp[0, :64 * 64].T.reshape(64, 64)
        dataset_phys[i, 1, :, :] = temp[0, 64 * 64:2 * 64 * 64].T.reshape(64, 64)
        dataset_phys[i, 2, :, :] = temp[0, 2 * 64 * 64:].T.reshape(64, 64)
    return torch.tensor(dataset_phys)

channel_final = 2048
# Define the model
class ConvAutoencoder(nn.Module):
    def __init__(self, num_latent=6, dropout_prob=0.4):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, channel_final//8, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(channel_final//8, channel_final//4, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(channel_final//4, channel_final//2, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(channel_final//2, channel_final, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(channel_final * (grid // 16) * (grid // 16), num_latent),
            
        )

        self.decoder_linear = nn.Sequential(
            
            nn.Linear(num_latent, channel_final * (grid // 16) * (grid // 16))
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(channel_final, channel_final//2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),

            nn.ConvTranspose2d(channel_final//2, channel_final//4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),

            nn.ConvTranspose2d(channel_final//4, channel_final//8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.Dropout2d(p=dropout_prob),

            nn.ConvTranspose2d(channel_final//8, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        if method=='tunable':
                self.a = nn.Parameter(torch.zeros(num_latent))
                self.b = nn.Parameter(torch.zeros(3))

    def forward(self, x):

        if method=='ae':
            x = self.encoder_conv(x)  # Apply convolutional block
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.encoder_linear(x)  # Apply linear block
        elif method=='naive':
            encoded_pod = x.reshape(x.shape[0], -1) @ vhr.T
            x = self.encoder_conv(x)  # Apply convolutional block
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.encoder_linear(x)  # Apply linear block
            x = x + encoded_pod
        elif method=='tunable':
            encoded_pod = x.reshape(x.shape[0], -1) @ vhr.T
            x = self.encoder_conv(x)  # Apply convolutional block
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.encoder_linear(x)  # Apply linear block
            x = self.a * x + (1-self.a) * encoded_pod

        if method=='ae':
            x = self.decoder_linear(x)  # Apply linear block
            x = x.reshape(x.shape[0], channel_final, grid//16, grid//16)  # Reshape to match the convolutional block input
            x = self.decoder_conv(x)
        elif method=='naive':
            decoded_pod = (x @ vhr).reshape(x.shape[0], 3, 64, 64)
            x = self.decoder_linear(x)  # Apply linear block
            x = x.reshape(x.shape[0], channel_final, grid//16, grid//16)  # Reshape to match the convolutional block input
            x = self.decoder_conv(x)
            x = x + decoded_pod
        elif method=='tunable':
            decoded_pod = (x @ vhr).reshape(x.shape[0], 3, 64, 64)
            x = self.decoder_linear(x)  # Apply linear block
            x = x.reshape(x.shape[0], channel_final, grid//16, grid//16)  # Reshape to match the convolutional block input
            x = self.decoder_conv(x)
            x = self.b.view(1,3,1,1) * x + (1 - self.b.view(1,3,1,1)) * decoded_pod
  

        return x

    def forward_encoder(self, x):


        if method=='ae':
            x = self.encoder_conv(x)  # Apply convolutional block
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.encoder_linear(x)  # Apply linear block
        elif method=='naive':
            encoded_pod = x.reshape(x.shape[0], -1) @ vhr.T
            x = self.encoder_conv(x)  # Apply convolutional block
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.encoder_linear(x)  # Apply linear block
            x = x + encoded_pod
        elif method=='tunable':
            encoded_pod = x.reshape(x.shape[0], -1) @ vhr.T
            x = self.encoder_conv(x)  # Apply convolutional block
            x = x.reshape(x.shape[0], -1)  # Flatten
            x = self.encoder_linear(x)  # Apply linear block
            x = self.a * x + (1-self.a) * encoded_pod

        return x
    def forward_decoder(self, x):

        if method=='ae':
            x = self.decoder_linear(x)  # Apply linear block
            x = x.reshape(x.shape[0], channel_final, grid//16, grid//16)  # Reshape to match the convolutional block input
            x = self.decoder_conv(x)
        elif method=='naive':
            decoded_pod = (x @ vhr).reshape(x.shape[0], 3, 64, 64)
            x = self.decoder_linear(x)  # Apply linear block
            x = x.reshape(x.shape[0], channel_final, grid//16, grid//16)  # Reshape to match the convolutional block input
            x = self.decoder_conv(x)
            x = x + decoded_pod
        elif method=='tunable':
            decoded_pod = (x @ vhr).reshape(x.shape[0], 3, 64, 64)
            x = self.decoder_linear(x)  # Apply linear block
            x = x.reshape(x.shape[0], channel_final, grid//16, grid//16)  # Reshape to match the convolutional block input
            x = self.decoder_conv(x)
            x = self.b.view(1,3,1,1) * x + (1 - self.b.view(1,3,1,1)) * decoded_pod

        return x

    # Hyperparameters
lrate = 3e-4
batch_size = 24
num_epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model, loss, optimizer
model = ConvAutoencoder(num_latent).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lrate)

# Prepare dataloaders
train_data = torch.utils.data.TensorDataset(swe_train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-3,
    cycle_momentum=False  # Disable momentum cycling for Adam
)

weights_filepath = f"best_weights_cae_{method}.pt"

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

#model.apply(init_weights)


# Training loop with saving weights
if mode == 'train':
    model.train()
    best_loss = float('inf')  # Initialize the best loss to infinity
    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the scheduler after each batch
            train_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = train_loss / len(train_loader)

        # Save weights if validation loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_filepath)
            print(f"Epoch {epoch + 1}: Improved loss to {best_loss:.6f}. Weights saved.")

        # Log training information
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
else:
    # Load saved weights for testing
    model.load_state_dict(torch.load(weights_filepath))
    print("Model weights loaded from", weights_filepath)# Visualization
model.eval()
time_window = 10
true_field = unscale(swe_test_data[time_window:time_window + 1], scaler)
pred_field = unscale(model(swe_test_data[time_window:time_window + 1].to(device)).cpu().detach(), scaler)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 12))
fields = [true_field, pred_field]
titles = [r'True $q_1$', r'True $q_2$', r'True $q_3$', r'Reconstructed $q_1$', r'Reconstructed $q_2$',
          r'Reconstructed $q_3$']

for i in range(2):
    for j in range(3):
        ax[i, j].imshow(fields[i][0, j].numpy(), cmap="viridis")
        ax[i, j].set_title(titles[i * 3 + j], fontsize=18)

plt.tight_layout()
plt.savefig(f'./recon_{method}.png')

true_field = unscale(swe_test_data, scaler)
pred_field = unscale(model(swe_test_data.to(device)).cpu().detach(), scaler)

np.save(f'./test_data_before_lstm_{method}.npy', true_field.cpu().numpy())
np.save(f'./test_prediction_before_lstm_{method}.npy', pred_field.cpu().numpy())
with torch.no_grad():
    encoded_list = []
    for i in range(90):
        encoded_list.append(model.forward_encoder(swe_train_data[100 * i:100 * (i + 1), :, :, :].to(device)).cpu().numpy())

encoded = np.array(encoded_list, dtype=np.float32)

parameters = np.load('../Equation_Based/Locations.npy')
parameters_train = parameters[:90]
parameters_test = parameters[90:]

lstm_training_data = np.copy(encoded)
num_train_snapshots = 90
total_size = np.shape(lstm_training_data)[0] * np.shape(lstm_training_data)[1]

time_window = 10
# Shape the inputs and outputs
input_seq = np.zeros(shape=(total_size - time_window * num_train_snapshots, time_window, num_latent + 2))
output_seq = np.zeros(shape=(total_size - time_window * num_train_snapshots, num_latent))

# Setting up inputs (window in, single timestep out)
sample = 0
for snapshot in range(num_train_snapshots):
    lstm_snapshot = lstm_training_data[snapshot, :, :]
    for t in range(time_window, 100):
        input_seq[sample, :, :num_latent] = lstm_snapshot[t - time_window:t, :]
        input_seq[sample, :, num_latent:] = parameters_train[snapshot, :]
        output_seq[sample, :] = lstm_snapshot[t, :]
        sample = sample + 1
# %%
# Explicitly adding parameter information to the encoded data for parameteric ROM
parameter_info = np.zeros(shape=(90, 100, 2), dtype='double')
# Setting up inputs
sample = 0
for snapshot in range(num_train_snapshots):
    parameter_info[snapshot, :, :] = parameters_train[snapshot, :]

total_training_data = np.concatenate((lstm_training_data, parameter_info), axis=-1)
total_training_data = np.concatenate((lstm_training_data, parameter_info), axis=-1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output for Dense layer
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

# Convert input and output sequences to PyTorch tensors
input_seq = torch.tensor(input_seq, dtype=torch.float32)
output_seq = torch.tensor(output_seq, dtype=torch.float32)

# Create DataLoader for batching
dataset = TensorDataset(input_seq, output_seq)
batch_size = 24
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
time_window = 10  # Timesteps
input_dim = 8  # Degrees of freedom (DOF)
hidden_dim = 50  # Number of hidden units in LSTM
output_dim = 6  # Number of latent space dimensions
num_layers = 3
num_epochs = 400
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)

def init_weights_lstm(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

# Apply initialization to your model
# lstm_model.apply(init_weights_lstm)

criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
scheduler = CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-2,
    cycle_momentum=False  # Disable momentum cycling for Adam
)
best_loss = float('inf')
lstm_filepath = f'lstm_weights_{method}.pt'

if mode=='train':
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = lstm_model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Average loss for this epoch
        avg_loss = train_loss / len(data_loader)

        # Save the best weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(lstm_model.state_dict(), lstm_filepath)
            print(f"Epoch {epoch + 1}: Improved loss to {best_loss:.6f}. Weights saved.")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
else:
    lstm_model.load_state_dict(torch.load(lstm_filepath))
    print("Model weights loaded from", lstm_filepath)  # Visualization

# Load best weights

lstm_model.eval()

# Assuming 'encoder', 'parameters_test', and 'swe_test_data' are already defined in PyTorch equivalents

# Encode the test data
encoded_list = []

with torch.no_grad():
    for i in range(10):
        test_batch = torch.tensor(swe_test_data[100 * i:100 * (i + 1), :, :, :], dtype=torch.float32).to(device)
        encoded_batch = model.forward_encoder(test_batch)  # Assuming 'model' has the encoder
        encoded_batch = encoded_batch.view(encoded_batch.size(0), -1)  # Flatten
        encoded_list.append(encoded_batch.cpu().numpy())

test_encoded = np.array(encoded_list)  # Convert to NumPy
lstm_testing_data = np.copy(test_encoded)

# Initialize input and output sequences
input_seq = torch.zeros((1, time_window, num_latent + 2), dtype=torch.float32).to(device)
output_seq_pred = np.zeros((10, 100, num_latent))

import time
# Time the simulation
start_time = time.time()

# Loop through simulations
for sim_num in range(10):
    # Set up initial inputs
    input_seq[0, :, :-2] = torch.tensor(lstm_testing_data[sim_num, :time_window, :], dtype=torch.float32).to(device)
    input_seq[0, :, -2] = torch.tensor(parameters_test[sim_num, 0], dtype=torch.float32)
    input_seq[0, :, -1] = torch.tensor(parameters_test[sim_num, 1], dtype=torch.float32)

    # Copy initial sequence to the output predictions
    output_seq_pred[sim_num, :time_window, :] = lstm_testing_data[sim_num, :time_window, :]

    for t in range(time_window, 100):
        # Predict next time step
        with torch.no_grad():
            next_pred = lstm_model(input_seq).cpu().numpy()  # Forward pass through LSTM

        # Update output sequence predictions
        output_seq_pred[sim_num, t, :] = next_pred[0, :]

        # Update input sequence for next prediction
        input_seq[0, :-1, :-2] = input_seq[0, 1:, :-2]  # Shift inputs left
        input_seq[0, -1, :-2] = torch.tensor(next_pred[0, :], dtype=torch.float32).to(device)  # Append new prediction

end_time = time.time()
print('Average time per simulation:', (end_time - start_time) / 10.0)

# Visualization
sim_num = 4  # Select one simulation for plotting

for i in range(num_latent):
    plt.figure(figsize=(7, 6))
    plt.plot(np.arange(100) * (0.5 / 100), lstm_testing_data[sim_num, :, i], 'r', label='True', linewidth=3)
    plt.plot(np.arange(100) * (0.5 / 100), output_seq_pred[sim_num, :, i], 'b--', label='Predicted', linewidth=3)

    if i == 0:
        plt.legend(fontsize=18)
    plt.ylabel('Latent space magnitude', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.tick_params(axis="both", labelsize=14)
    plt.savefig(f'LSTM_Sim_{sim_num}_Mode_{i}.png')
    # plt.show()
    plt.savefig(f'./latent_space_{method}_{i}.png')
# Assuming 'unscale' and 'scaler' are defined
swe_test_data_phys = unscale(swe_test_data, scaler)

# Initialize the array for scaled reconstruction
cae_test_scaled = np.empty(shape=(1000, 3, 64, 64))

# Decode and reconstruct the data
start_time = time.time()
with torch.no_grad():
    for sim_num in range(10):
        # Decode predictions
        decoded_valid = model.forward_decoder(torch.tensor(output_seq_pred[sim_num], dtype=torch.float32).to(device))
        # Convert decoded output to NumPy
        decoded_valid = decoded_valid.cpu().numpy()
        # Save reconstructed data
        cae_test_scaled[100 * sim_num:100 * (sim_num + 1)] = decoded_valid

# Unscale the predictions
cae_test_preds = unscale(torch.tensor(cae_test_scaled, dtype=torch.float32), scaler)
end_time = time.time()
cae_test_preds = np.transpose(cae_test_preds.numpy(), (0, 2,3,1))
print('Average time for reconstruction:', (end_time - start_time) / 10.0)

# Error calculation (excluding the initial condition)
q1_cae_error = 0.0
q2_cae_error = 0.0
q3_cae_error = 0.0
num_fields = 0

swe_test_data_phys = np.transpose(unscale(swe_test_data, scaler).numpy(), (0, 2,3,1))
# Calculate errors
for sim_num in range(10):
    for i in range(1, 100):  # Skip the initial condition
        q1_cae_error += np.mean(
            (cae_test_preds[sim_num * 100 + i, :, :, 0] - swe_test_data_phys[sim_num * 100 + i, :, :, 0]) ** 2)
        q2_cae_error += np.mean(
            (cae_test_preds[sim_num * 100 + i, :, :, 1] - swe_test_data_phys[sim_num * 100 + i, :, :, 1]) ** 2)
        q3_cae_error += np.mean(
            (cae_test_preds[sim_num * 100 + i, :, :, 2] - swe_test_data_phys[sim_num * 100 + i, :, :, 2]) ** 2)

        num_fields += 1

# Compute average errors
q1_cae_error = q1_cae_error / num_fields
q2_cae_error = q2_cae_error / num_fields
q3_cae_error = q3_cae_error / num_fields

# Print results
print(f"Q1 CAE Error: {q1_cae_error}")
print(f"Q2 CAE Error: {q2_cae_error}")
print(f"Q3 CAE Error: {q3_cae_error}")

def gp_loader(num_dof):
    q1_gp_coeffs = np.load('../Equation_Based/'  + '/PCA_Coefficients_q1_pred.npy')
    q2_gp_coeffs = np.load('../Equation_Based/'  + '/PCA_Coefficients_q2_pred.npy')
    q3_gp_coeffs = np.load('../Equation_Based/'  + '/PCA_Coefficients_q3_pred.npy')

    q1_modes = np.load('../Equation_Based/'  + '/PCA_Vectors_q1.npy')
    q2_modes = np.load('../Equation_Based/'  + '/PCA_Vectors_q2.npy')
    q3_modes = np.load('../Equation_Based/'  + '/PCA_Vectors_q3.npy')

    try:
        q1_gp = np.matmul(q1_modes, q1_gp_coeffs)
        q2_gp = np.matmul(q2_modes, q2_gp_coeffs)
        q3_gp = np.matmul(q3_modes, q3_gp_coeffs)
    except:
        q1_gp = np.matmul(q1_modes[:, :num_dof], q1_gp_coeffs)
        q2_gp = np.matmul(q2_modes[:, :num_dof], q2_gp_coeffs)
        q3_gp = np.matmul(q3_modes[:, :num_dof], q3_gp_coeffs)

    gp_test_data = np.zeros(shape=(np.shape(swe_test_data)[0], 64, 64, 3))  # Channels last
    for i in range(np.shape(swe_test_data)[0]):
        temp_1 = q1_gp[:64 * 64, i].reshape(64, 64).T
        temp_2 = q2_gp[:64 * 64, i].reshape(64, 64).T
        temp_3 = q3_gp[:64 * 64, i].reshape(64, 64).T
        gp_test_data[i, :, :, 0] = np.transpose(temp_1[:, :])
        gp_test_data[i, :, :, 1] = np.transpose(temp_2[:, :])
        gp_test_data[i, :, :, 2] = np.transpose(temp_3[:, :])

    # Error calculation for GP (except initial condition)
    q1_gp_error = 0
    q2_gp_error = 0
    q3_gp_error = 0

    num_fields = 0
    for sim_num in range(10):
        for i in range(1, 100):
            q1_gp_error += np.mean(
                (gp_test_data[sim_num * 100 + i, :, :, 0] - swe_test_data_phys[sim_num * 100 + i, :, :, 0]) ** 2)
            q2_gp_error += np.mean(
                (gp_test_data[sim_num * 100 + i, :, :, 1] - swe_test_data_phys[sim_num * 100 + i, :, :, 1]) ** 2)
            q3_gp_error += np.mean(
                (gp_test_data[sim_num * 100 + i, :, :, 2] - swe_test_data_phys[sim_num * 100 + i, :, :, 2]) ** 2)

            num_fields += 1

    q1_gp_error = q1_gp_error / num_fields
    q2_gp_error = q2_gp_error / num_fields
    q3_gp_error = q3_gp_error / num_fields

    return gp_test_data, q1_gp_error, q2_gp_error, q3_gp_error


# %%
# gp_test_data_4, q1_gp_error_4, q2_gp_error_4, q3_gp_error_4 = gp_loader(4)
gp_test_data_6, q1_gp_error_6, q2_gp_error_6, q3_gp_error_6 = gp_loader(6)
# gp_test_data_40, q1_gp_error_40, q2_gp_error_40, q3_gp_error_40 = gp_loader(40)
# %%
time = 1

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 12))

cs1 = ax[0, 0].imshow(swe_test_data_phys[time, :, :, 0], label='Truth', vmin=0.85, vmax=1.15)
cs2 = ax[1, 0].imshow(cae_test_preds[time, :, :, 0], label='CAE-LSTM', vmin=0.85, vmax=1.15)
cs3 = ax[2, 0].imshow(gp_test_data_6[time, :, :, 0], label='POD-GP 6 modes', vmin=0.85, vmax=1.15)
# cs4 = ax[3, 0].imshow(gp_test_data_40[time, :, :, 0], label='POD-GP 40 modes', vmin=0.85, vmax=1.15)

fig.colorbar(cs1, ax=ax[0, 0], fraction=0.046, pad=0.04)
fig.colorbar(cs2, ax=ax[1, 0], fraction=0.046, pad=0.04)
fig.colorbar(cs3, ax=ax[2, 0], fraction=0.046, pad=0.04)
# fig.colorbar(cs4, ax=ax[3, 0], fraction=0.046, pad=0.04)

cs1 = ax[0, 1].imshow(swe_test_data_phys[time, :, :, 1], label='Truth', vmin=-0.3, vmax=0.2)
cs2 = ax[1, 1].imshow(cae_test_preds[time, :, :, 1], label='Prediction', vmin=-0.3, vmax=0.2)
cs3 = ax[2, 1].imshow(gp_test_data_6[time, :, :, 1], label='POD-GP 6 modes', vmin=-0.3, vmax=0.2)
# cs4 = ax[3, 1].imshow(gp_test_data_40[time, :, :, 1], label='POD-GP 40 modes', vmin=-0.3, vmax=0.2)

fig.colorbar(cs1, ax=ax[0, 1], fraction=0.046, pad=0.04)
fig.colorbar(cs2, ax=ax[1, 1], fraction=0.046, pad=0.04)
fig.colorbar(cs3, ax=ax[2, 1], fraction=0.046, pad=0.04)
# fig.colorbar(cs4, ax=ax[3, 1], fraction=0.046, pad=0.04)

cs1 = ax[0, 2].imshow(swe_test_data_phys[time, :, :, 2], label='Truth', vmin=-0.5, vmax=0.5)
cs2 = ax[1, 2].imshow(cae_test_preds[time, :, :, 2], label='Prediction', vmin=-0.5, vmax=0.5)
cs3 = ax[2, 2].imshow(gp_test_data_6[time, :, :, 2], label='POD-GP 6 modes', vmin=-0.5, vmax=0.5)
# cs4 = ax[3, 2].imshow(gp_test_data_40[time, :, :, 2], label='POD-GP 40 modes', vmin=-0.5, vmax=0.5)

fig.colorbar(cs1, ax=ax[0, 2], fraction=0.046, pad=0.04)
fig.colorbar(cs2, ax=ax[1, 2], fraction=0.046, pad=0.04)
fig.colorbar(cs3, ax=ax[2, 2], fraction=0.046, pad=0.04)
# fig.colorbar(cs4, ax=ax[3, 2], fraction=0.046, pad=0.04)

for i in range(3):
    for j in range(3):
        ax[i, j].set_xlabel('x', fontsize=18)
        ax[i, j].set_ylabel('y', fontsize=18)
        ax[i, j].tick_params(labelsize=14)

ax[0, 0].set_title(r'True $q_1$', fontsize=18)
ax[0, 1].set_title(r'True $q_2$', fontsize=18)
ax[0, 2].set_title(r'True $q_3$', fontsize=18)

ax[1, 0].set_title(r'CAE-LSTM', fontsize=18)
ax[1, 1].set_title(r'CAE-LSTM', fontsize=18)
ax[1, 2].set_title(r'CAE-LSTM', fontsize=18)

ax[2, 0].set_title(r'POD-GP (6 modes)', fontsize=18)
ax[2, 1].set_title(r'POD-GP (6 modes)', fontsize=18)
ax[2, 2].set_title(r'POD-GP (6 modes)', fontsize=18)

# ax[3, 0].set_title(r'POD-GP (40 modes)', fontsize=18)
# ax[3, 1].set_title(r'POD-GP (40 modes)', fontsize=18)
# ax[3, 2].set_title(r'POD-GP (40 modes)', fontsize=18)

plt.subplots_adjust(wspace=-0.5, hspace=0.5)
plt.tight_layout()
plt.savefig(f'CAE_GP_Comparison_{method}_0.png')
# plt.show()
# %%
time = 200

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 12))
vmin0 = np.min(swe_test_data_phys[time, :, :, 0])
vmax0 = np.max(swe_test_data_phys[time, :, :, 0])
cs1 = ax[0, 0].imshow(swe_test_data_phys[time, :, :, 0], label='Truth')
cs2 = ax[1, 0].imshow(cae_test_preds[time, :, :, 0], label='CAE-LSTM')
cs3 = ax[2, 0].imshow(gp_test_data_6[time, :, :, 0], label='POD-GP 6 modes')
# cs4 = ax[3, 0].imshow(gp_test_data_40[time, :, :, 0], label='POD-GP 40 modes', vmin=1.0, vmax=1.1)

fig.colorbar(cs1, ax=ax[0, 0], fraction=0.046, pad=0.04)
fig.colorbar(cs2, ax=ax[1, 0], fraction=0.046, pad=0.04)
fig.colorbar(cs3, ax=ax[2, 0], fraction=0.046, pad=0.04)
# fig.colorbar(cs4, ax=ax[3, 0], fraction=0.046, pad=0.04)

vmin1 = np.min(swe_test_data_phys[time, :, :, 1])
vmax1 = np.max(swe_test_data_phys[time, :, :, 1])
cs1 = ax[0, 1].imshow(swe_test_data_phys[time, :, :, 1], label='Truth')
cs2 = ax[1, 1].imshow(cae_test_preds[time, :, :, 1], label='Prediction')
cs3 = ax[2, 1].imshow(gp_test_data_6[time, :, :, 1], label='POD-GP 6 modes')
# cs4 = ax[3, 1].imshow(gp_test_data_40[time, :, :, 1], label='POD-GP 40 modes', vmin=-0.1, vmax=0.1)

fig.colorbar(cs1, ax=ax[0, 1], fraction=0.046, pad=0.04)
fig.colorbar(cs2, ax=ax[1, 1], fraction=0.046, pad=0.04)
fig.colorbar(cs3, ax=ax[2, 1], fraction=0.046, pad=0.04)
# fig.colorbar(cs4, ax=ax[3, 1], fraction=0.046, pad=0.04)

vmin2 = np.min(swe_test_data_phys[time, :, :, 2])
vmax2 = np.max(swe_test_data_phys[time, :, :, 2])
cs1 = ax[0, 2].imshow(swe_test_data_phys[time, :, :, 2], label='Truth')
cs2 = ax[1, 2].imshow(cae_test_preds[time, :, :, 2], label='Prediction')
cs3 = ax[2, 2].imshow(gp_test_data_6[time, :, :, 2], label='POD-GP 6 modes')
# cs4 = ax[3, 2].imshow(gp_test_data_40[time, :, :, 2], label='POD-GP 40 modes', vmin=-0.1, vmax=0.1)

fig.colorbar(cs1, ax=ax[0, 2], fraction=0.046, pad=0.04)
fig.colorbar(cs2, ax=ax[1, 2], fraction=0.046, pad=0.04)
fig.colorbar(cs3, ax=ax[2, 2], fraction=0.046, pad=0.04)
# fig.colorbar(cs4, ax=ax[3, 2], fraction=0.046, pad=0.04)

for i in range(3):
    for j in range(3):
        ax[i, j].set_xlabel('x', fontsize=18)
        ax[i, j].set_ylabel('y', fontsize=18)
        ax[i, j].tick_params(labelsize=14)

ax[0, 0].set_title(r'True $q_1$', fontsize=18)
ax[0, 1].set_title(r'True $q_2$', fontsize=18)
ax[0, 2].set_title(r'True $q_3$', fontsize=18)

ax[1, 0].set_title(r'CAE-LSTM $q_1$', fontsize=18)
ax[1, 1].set_title(r'CAE-LSTM $q_2$', fontsize=18)
ax[1, 2].set_title(r'CAE-LSTM $q_3$', fontsize=18)

ax[2, 0].set_title(r'POD-GP $q_1$ (6 modes)', fontsize=18)
ax[2, 1].set_title(r'POD-GP $q_2$ (6 modes)', fontsize=18)
ax[2, 2].set_title(r'POD-GP $q_3$ (6 modes)', fontsize=18)

# ax[3, 0].set_title(r'POD-GP $q_1$ (40 modes)', fontsize=18)
# ax[3, 1].set_title(r'POD-GP $q_2$ (40 modes)', fontsize=18)
# ax[3, 2].set_title(r'POD-GP $q_3$ (40 modes)', fontsize=18)

plt.subplots_adjust(wspace=-0.5, hspace=0.5)
plt.tight_layout()
plt.savefig(f'CAE_GP_Comparison_{method}_1.png')
# plt.show()
# %%
print('MSE Test ', np.mean((swe_test_data_phys - cae_test_preds)**2))
np.save(f'./test_data_after_lstm_{method}.npy', swe_test_data_phys.transpose(0,3,1,2))
np.save(f'./test_prediction_after_lstm_{method}.npy', cae_test_preds.transpose(0,3,1,2))
np.save(f'./test_error_{method}.npy', np.mean((swe_test_data_phys - cae_test_preds)**2))
print('CAE error for $q_1$:', q1_cae_error, '6 mode GP error for $q_1$:', q1_gp_error_6)
print('CAE error for $q_2$:', q2_cae_error, '6 mode GP error for $q_2$:', q2_gp_error_6)
print('CAE error for $q_3$:', q3_cae_error, '6 mode GP error for $q_3$:', q3_gp_error_6)

print('\n')
