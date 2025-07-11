import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# np.random.seed(1990)
# torch.manual_seed(1990)
import torch.nn.functional as F
import os
import h5py

# Initialize argument parser
parser = argparse.ArgumentParser(description='Autoencoder Arguments')

# Add arguments
parser.add_argument('--grid', type=int, default=32, help='Grid dimension')
parser.add_argument('--encoded_space_dim', type=int, default=6, help='Dimension of the encoded space')
parser.add_argument('--method', type=str, default='pod-ae-tunable', choices=['ae', 'pod', 'pod-ae', 'pod-ae-tunable', 'naive'],
                    help='Method for autoencoder')
parser.add_argument('--seed', type=int, default=1990,
                    help='Random seed')
parser.add_argument('--epochs', type=int, default=40000,
                    help='epochs')

# Parse arguments
args = parser.parse_args()

# Access arguments
grid = args.grid
encoded_space_dim = args.encoded_space_dim
method = args.method
seed = args.seed
epochs = args.epochs

rank = encoded_space_dim
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setup
# torch.backends.cudnn.deterministic = True

import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, dropout_prob=0.4):
        super(Encoder, self).__init__()
        self.enc1 = nn.Conv3d(3, 256, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.drop1 = nn.Dropout3d(p=dropout_prob)
        self.enc2 = nn.Conv3d(256, 512, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.drop2 = nn.Dropout3d(p=dropout_prob)
        self.enc3 = nn.Conv3d(512, 1024, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.drop3 = nn.Dropout3d(p=dropout_prob)
        self.enc4 = nn.Conv3d(1024, 2048, kernel_size=5, stride=2, padding=2, padding_mode='circular')
        self.drop4 = nn.Dropout3d(p=dropout_prob)
  
        self.fc = nn.Sequential(
           nn.Linear(2048 * (grid // 16) * (grid // 16) * (grid // 16), encoded_space_dim),
  
        )

    def forward(self, x):
        x = F.silu(self.enc1(x))
        x = self.drop1(x)
        x = F.silu(self.enc2(x))
        x = self.drop2(x)
        x = F.silu(self.enc3(x))
        x = self.drop3(x)
        x = F.silu(self.enc4(x))
        x = self.drop4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, dropout_prob=0.4):
        super(Decoder, self).__init__()
   
        self.fc = nn.Sequential(
   
           nn.Linear(encoded_space_dim, 2048 * (grid // 16) * (grid // 16) * (grid // 16))
        )
        self.dec1 = nn.ConvTranspose3d(2048, 1024, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.drop1 = nn.Dropout3d(p=dropout_prob)
        self.dec2 = nn.ConvTranspose3d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.drop2 = nn.Dropout3d(p=dropout_prob)
        self.dec3 = nn.ConvTranspose3d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.drop3 = nn.Dropout3d(p=dropout_prob)
        self.dec4 = nn.ConvTranspose3d(256, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 2048, (grid // 16), (grid // 16), (grid // 16))
        x = F.silu(self.dec1(x))
        x = self.drop1(x)
        x = F.silu(self.dec2(x))
        x = self.drop2(x)
        x = F.silu(self.dec3(x))
        x = self.drop3(x)
        x = self.dec4(x)
        return x


class AutoEncoder():
    def __init__(self, rank, method, normalize=True):
        self.r = rank
        self.method = method
        self.normalize = normalize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_svd(self, X):
        
        blah = np.arange(X.shape[0])
        np.random.shuffle(blah)
        X = X[blah]
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        vhr = torch.from_numpy(vh[:self.r, :]).to(self.device)

        return vhr

    def _init_encoder_decoder(self, X):
        if self.method == 'pod':
            params = torch.load(f'3d_cnn_{method}_{grid}_{encoded_space_dim}_{seed}/{method}_parameters.pth')
            self.vhr = torch.tensor(params['vhr_matrix'], dtype=torch.float).to(self.device)
            self.encoder = lambda x: torch.matmul(x, self.vhr.T)
            self.decoder = lambda x: torch.matmul(x, self.vhr)

        elif self.method == 'ae':
            self.encoder_mlp = Encoder(self.r).to(self.device)
            self.decoder_mlp = Decoder(self.r).to(self.device)
            if torch.cuda.device_count() > 1:
                self.encoder_mlp = nn.DataParallel(self.encoder_mlp)
                self.decoder_mlp = nn.DataParallel(self.decoder_mlp)

            params = torch.load(f'3d_cnn_{method}_{grid}_{encoded_space_dim}_{seed}/{method}_parameters.pth')
            self.encoder_mlp.load_state_dict(params['encoder_mlp_state_dict'])
            self.decoder_mlp.load_state_dict(params['decoder_mlp_state_dict'])
            # self.encoder = la x: decoder_mlp(x)mbda x: encoder_mlp(x)
            # self.decoder = lambda
            self.ParameterList = [*self.encoder_mlp.parameters(), *self.decoder_mlp.parameters()]

        elif self.method == 'pod-ae-tunable':
            self.encoder_mlp = Encoder(self.r).to(self.device)
            self.decoder_mlp = Decoder(self.r).to(self.device)
            params = torch.load(f'3d_cnn_{method}_{grid}_{encoded_space_dim}_{seed}/{method}_parameters.pth')
            self.vhr = torch.tensor(params['vhr_matrix'], dtype=torch.float).to(self.device)
            self.encoder_mlp.load_state_dict(params['encoder_mlp_state_dict'])
            self.decoder_mlp.load_state_dict(params['decoder_mlp_state_dict'])
            self.encoder_pod = lambda x: torch.matmul(x, self.vhr.T)
            self.decoder_pod = lambda x: torch.matmul(x, self.vhr)
            if torch.cuda.device_count() > 1:
                self.encoder_mlp = nn.DataParallel(self.encoder_mlp)
                self.decoder_mlp = nn.DataParallel(self.decoder_mlp)
            self.a = torch.tensor(params['a_parameter'], dtype=torch.float).to(self.device)
            self.b = torch.tensor(params['b_parameter'], dtype=torch.float).to(self.device)

            self.ParameterList = [*self.encoder_mlp.parameters(), *self.decoder_mlp.parameters()]

        elif self.method == 'naive':
            self.encoder_mlp = Encoder(self.r).to(self.device)
            self.decoder_mlp = Decoder(self.r).to(self.device)
            params = torch.load(f'3d_cnn_{method}_{grid}_{encoded_space_dim}_{seed}/{method}_parameters.pth')
            self.vhr = torch.tensor(params['vhr_matrix'], dtype=torch.float).to(self.device)
            self.encoder_mlp.load_state_dict(params['encoder_mlp_state_dict'])
            self.decoder_mlp.load_state_dict(params['decoder_mlp_state_dict'])
            self.encoder_pod = lambda x: torch.matmul(x, self.vhr.T)
            self.decoder_pod = lambda x: torch.matmul(x, self.vhr)
            if torch.cuda.device_count() > 1:
                self.encoder_mlp = nn.DataParallel(self.encoder_mlp)
                self.decoder_mlp = nn.DataParallel(self.decoder_mlp)
            self.ParameterList = [*self.encoder_mlp.parameters(), *self.decoder_mlp.parameters()]
        else:
            print('Unknown Method')

    def _eval(self, data):

        data = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.loss_fn = nn.MSELoss()
        if self.method == 'pod':
            with torch.no_grad():

                encoded = self.encoder(data.reshape(data.shape[0], -1))
                decoded = self.decoder(encoded).reshape(data.shape[0], *data.shape[1:])

            decoded_unnorm = decoded.detach().cpu().numpy() * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm-data_unnorm)**2)


        if self.method == 'ae':
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            with torch.no_grad():
                    encoded = self.encoder_mlp(data)
                    decoded = self.decoder_mlp(encoded)

            decoded_unnorm = decoded.detach().cpu().numpy() * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm- data_unnorm)**2)

        if self.method == 'pod-ae-tunable':
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            with torch.no_grad():
                encoded = (1.0 - self.a) * self.encoder_pod(
                    data.reshape(data.shape[0], -1)) + self.a * self.encoder_mlp(data)
                decoded = (1.0 - self.b.view(1, 3, 1, 1, 1)) * self.decoder_pod(encoded).reshape(
                    encoded.shape[0],
                    *data.shape[
                     1:]) + self.b.view(1, 3, 1, 1, 1) * self.decoder_mlp(
                    encoded)
            decoded_unnorm = decoded.detach().cpu().numpy() * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm-data_unnorm)**2)

        if self.method == 'naive':
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            with torch.no_grad():
                encoded = self.encoder_pod(
                    data.reshape(data.shape[0], -1)) + self.encoder_mlp(data)
                decoded = self.decoder_pod(encoded).reshape(
                    encoded.shape[0],
                    *data.shape[
                     1:]) + self.decoder_mlp(
                    encoded)
            decoded_unnorm = decoded.detach().cpu().numpy() * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm-data_unnorm)**2)

        return encoded, decoded, decoded_unnorm, data_unnorm, loss

    def _decode(self, encoded, data):

        encoded = torch.tensor(encoded, dtype=torch.float32, device=self.device)
        chunks = 101
        if encoded.shape[0]>batch_size:
            chunks = torch.chunk(encoded, chunks)
        self.loss_fn = nn.MSELoss()
        if self.method == 'pod':
            with torch.no_grad():
                decoded = self.decoder(encoded).reshape(data.shape[0], *data.shape[1:])

            decoded_unnorm = decoded.detach().cpu().numpy() * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm-data_unnorm)**2)


        if self.method == 'ae':
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            decoded = []
            with torch.no_grad():
                for chunk in chunks:
                    decoded_chunk = self.decoder_mlp(chunk)
                    decoded.append(decoded_chunk.cpu().numpy())

            decoded = np.concatenate(decoded, axis=0)
            decoded_unnorm = decoded * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm- data_unnorm)**2)

        if self.method == 'pod-ae-tunable':
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            decoded = []
            with torch.no_grad():
                for chunk in chunks:
                    decoded_chunk = (1.0 - self.b.view(1, 3, 1, 1, 1)) * self.decoder_pod(chunk).reshape(
                        chunk.shape[0],
                        *data.shape[
                         1:]) + self.b.view(1, 3, 1, 1, 1) * self.decoder_mlp(
                        chunk)
                    decoded.append(decoded_chunk.cpu().numpy())
            decoded = np.concatenate(decoded, axis=0)
            decoded_unnorm = decoded * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm-data_unnorm)**2)

        if self.method == 'naive':
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            decoded = []
            with torch.no_grad():
                for chunk in chunks:
                    decoded_chunk = self.decoder_pod(chunk).reshape(
                    chunk.shape[0],
                    *data.shape[
                     1:]) + self.decoder_mlp(
                    chunk)
                    decoded.append(decoded_chunk.cpu().numpy())

            decoded = np.concatenate(decoded, axis=0)
            decoded_unnorm = decoded * std_value + mean_value
            data_unnorm = data.detach().cpu().numpy() * std_value + mean_value
            loss = np.mean((decoded_unnorm-data_unnorm)**2)

        return encoded, decoded, decoded_unnorm, data_unnorm, loss




save_path = './3d_cnn_' + method + '_' + str(grid) + '_' + str(encoded_space_dim) + '_' + str(seed)

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Folder '{save_path}' created.")
else:
    print(f"Folder '{save_path}' already exists.")

if not os.path.exists(save_path + '/figures'):
    os.makedirs(save_path + '/figures')
    print(f"Folder '{save_path + '/figures'}' created.")
else:
    print(f"Folder '{save_path + '/figures'}' already exists.")

velocity_data = np.load('./train_data_multi_traj.npy').reshape(50 * 101, 3, 32, 32, 32)
velocity_data = np.float32(velocity_data)
num_samples = velocity_data.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Define split size
split_index = int(0.7 * num_samples)

X_train = np.float32(np.load('./train_data_multi_traj.npy')[:50, :50].reshape(50*50, 3, grid , grid , grid))
X_test = np.float32(np.load('./train_data_multi_traj.npy')[:50, 50:].reshape(50*51, 3, grid , grid , grid))

mean_value = np.mean(X_train, axis=0)
std_value = np.std(X_train, axis=0)

X_train_normalized = (X_train - mean_value) / (std_value)
X_test_normalized = (X_test - mean_value) / (std_value)

lstm_train_data_norm = X_train_normalized.reshape(50, 50, 3, 32, 32, 32)
lstm_test_data_norm = X_test_normalized.reshape(50, 51, 3, 32, 32, 32)


ae = AutoEncoder(rank, method, True)
ae._init_encoder_decoder(X_train_normalized)

encoded_lstm_train = []
encoded_lstm_test = []
for i in range(lstm_train_data_norm.shape[0]):
    encoded, decoded, decoded_unnorm, data_unnorm, loss = ae._eval(lstm_train_data_norm[i])
    encoded_lstm_train.append(encoded.cpu().numpy())

for i in range(lstm_test_data_norm.shape[0]):
    encoded, decoded, decoded_unnorm, data_unnorm, loss = ae._eval(lstm_test_data_norm[i])
    encoded_lstm_test.append(encoded.cpu().numpy())

batch_size = 24
encoded_lstm_train = np.array(encoded_lstm_train, dtype=np.float32)
encoded_lstm_test = np.array(encoded_lstm_test, dtype=np.float32)

encoded, decoded, decoded_unnorm, data_unnorm, loss = ae._decode(encoded_lstm_test.reshape(50*51, encoded_space_dim), torch.tensor(lstm_test_data_norm.reshape(50*51, 3, 32, 32, 32), dtype=torch.float32, device='cuda'))

print('before lstm recon error ', np.mean((data_unnorm-decoded_unnorm)**2))
np.save(f'./loss_before_lstm_{method}_{encoded_space_dim}.npy', np.mean((data_unnorm-decoded_unnorm)**2))
np.save(f'./test_data_before_lstm_{method}.npy', data_unnorm)
np.save(f'./test_prediction_before_lstm_{method}.npy', decoded_unnorm)
mean_encoded = np.mean(encoded_lstm_train, axis=(0,1))
std_encoded = np.std(encoded_lstm_train, axis=(0,1))
encoded_lstm_train = (encoded_lstm_train - mean_encoded)/std_encoded
encoded_lstm_test = (encoded_lstm_test - mean_encoded)/std_encoded

num_train_snapshots = encoded_lstm_train.shape[0]
total_size = encoded_lstm_train.shape[0] * encoded_lstm_train.shape[1]

time_window = 10
# Shape the inputs and outputs
input_seq = np.zeros(shape=(total_size - time_window * num_train_snapshots, time_window, encoded_space_dim))
output_seq = np.zeros(shape=(total_size - time_window * num_train_snapshots, encoded_space_dim))

# Setting up inputs (window in, single timestep out)
sample = 0
for snapshot in range(num_train_snapshots):
    lstm_snapshot = encoded_lstm_train[snapshot, :, :]
    for t in range(time_window, 50):
        input_seq[sample, :, :encoded_space_dim] = lstm_snapshot[t - time_window:t, :]
        output_seq[sample, :] = lstm_snapshot[t, :]
        sample = sample + 1

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.6):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output for Dense layer
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

input_seq = torch.tensor(input_seq, dtype=torch.float32)
output_seq = torch.tensor(output_seq, dtype=torch.float32)

# Create DataLoader for batching
dataset = TensorDataset(input_seq, output_seq)
batch_size = 24
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

time_window = 10  # Timesteps
input_dim = encoded_space_dim  # Degrees of freedom (DOF)
hidden_dim = 128  # Number of hidden units in LSTM
output_dim = encoded_space_dim  # Number of latent space dimensions
num_layers = 3
num_epochs = 200
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
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-2,
    cycle_momentum=False  # Disable momentum cycling for Adam
)
best_loss = float('inf')
lstm_filepath = f'lstm_weights_{method}_{encoded_space_dim}.pt'
mode = 'test'

def test_lstm(encoded_lstm, lstm_data_norm):
    lstm_model.eval()

    # Initialize input and output sequences
    input_seq = torch.zeros((1, time_window, encoded_space_dim), dtype=torch.float32).to(device)
    output_seq_pred = np.zeros((encoded_lstm.shape[0], encoded_lstm.shape[1], encoded_space_dim))

    # Loop through simulations
    for sim_num in range(encoded_lstm.shape[0]):
        # Set up initial inputs
        input_seq[0, :, :] = torch.tensor(encoded_lstm[sim_num, :time_window, :], dtype=torch.float32).to(device)

        # Copy initial sequence to the output predictions
        output_seq_pred[sim_num, :time_window, :] = encoded_lstm[sim_num, :time_window, :]

        for t in range(time_window, encoded_lstm.shape[1]):
            # Predict next time step
            with torch.no_grad():
                next_pred = lstm_model(input_seq).cpu().numpy()  # Forward pass through LSTM

            # Update output sequence predictions
            output_seq_pred[sim_num, t, :] = next_pred[0, :]

            # Update input sequence for next prediction
            input_seq[0, :-1, :] = input_seq[0, 1:, :].clone()  # Shift inputs left
            input_seq[0, -1, :] = torch.tensor(next_pred[0, :], dtype=torch.float32).to(device)

    output_seq_pred = output_seq_pred * std_encoded + mean_encoded

    encoded, decoded, decoded_unnorm, data_unnorm, loss = ae._decode(output_seq_pred.reshape(output_seq_pred.shape[0] * output_seq_pred.shape[1], encoded_space_dim),
                                                                     torch.tensor(lstm_data_norm.reshape(lstm_data_norm.shape[0] * lstm_data_norm.shape[1], lstm_data_norm.shape[2],
                                                                                                         lstm_data_norm.shape[3], lstm_data_norm.shape[4], lstm_data_norm.shape[5]),
                                                                                                          dtype=torch.float32, device=device))

    return encoded, decoded, decoded_unnorm, data_unnorm, loss

if mode=='train':
    for epoch in range(num_epochs):

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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")
        # Save the best weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            # _, _, _, _, train_recon_loss = test_lstm(encoded_lstm_train, lstm_train_data_norm)
            # _, _, _, _, test_recon_loss = test_lstm(encoded_lstm_test, lstm_test_data_norm)
            torch.save(lstm_model.state_dict(), lstm_filepath)
            print(f"Epoch {epoch + 1}: Improved loss to {best_loss}. Weights saved.")
            # _, _, _, _, train_recon_loss = test_lstm(encoded_lstm_train, lstm_train_data_norm[0])
            # _, _, _, _, test_recon_loss = test_lstm(encoded_lstm_test, lstm_test_data_norm)
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Recon Train Loss: {train_recon_loss}, Recon Test Loss: {test_recon_loss}")
            lstm_model.train()

 # Visualization

lstm_model.load_state_dict(torch.load(lstm_filepath))
print("Model weights loaded from", lstm_filepath)
lstm_model.eval()

# Initialize input and output sequences
input_seq = torch.zeros((1, time_window, encoded_space_dim), dtype=torch.float32).to(device)
output_seq_pred = np.zeros((50, 51, encoded_space_dim))

# Loop through simulations
for sim_num in range(50):
    # Set up initial inputs
    input_seq[0, :, :] = torch.tensor(encoded_lstm_test[sim_num, :time_window, :], dtype=torch.float32).to(device)

    # Copy initial sequence to the output predictions
    output_seq_pred[sim_num, :time_window, :] = encoded_lstm_test[sim_num, :time_window, :]

    for t in range(time_window, 51):
        # Predict next time step
        with torch.no_grad():
            next_pred = lstm_model(input_seq).cpu().numpy()  # Forward pass through LSTM

        # Update output sequence predictions
        output_seq_pred[sim_num, t, :] = next_pred[0, :]

        # Update input sequence for next prediction
        input_seq[0, :-1, :] = input_seq[0, 1:, :].clone()  # Shift inputs left
        input_seq[0, -1, :] = torch.tensor(next_pred[0, :], dtype=torch.float32).to(device)  # Append new prediction

output_seq_pred = output_seq_pred  * std_encoded + mean_encoded

encoded, decoded, decoded_unnorm, data_unnorm, loss = ae._decode(output_seq_pred.reshape(50*51, encoded_space_dim), torch.tensor(lstm_test_data_norm.reshape(50*51, 3, 32, 32, 32), dtype=torch.float32, device=device))

np.save(f'./loss_after_lstm_{method}_{encoded_space_dim}.npy', loss)
np.save(f'./test_prediction_after_lstm_{method}.npy', decoded_unnorm)
np.save(f'./test_data_after_lstm_{method}.npy', data_unnorm)
print('after lstm test recon error ', np.mean((decoded_unnorm-data_unnorm)**2))
# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
vmin = data_unnorm[1848, 1, :, 29, :].min()
vmax = data_unnorm[1848, 1, :, 29, :].max()
# First subplot: Sine Wave
axes[0].imshow(data_unnorm[1848, 1, :, 29, :], cmap='viridis', vmin=vmin, vmax = vmax)
axes[0].set_title("GT")

# Second subplot: Cosine Wave
axes[1].imshow(decoded_unnorm[1848, 1, :, 29, :], cmap='viridis', vmin=vmin, vmax = vmax)
axes[1].set_title("Model Prediction")


# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig(f'./comparison_after_lstm_{method}_{encoded_space_dim}.png')
plt.show()


# Initialize input and output sequences
input_seq = torch.zeros((1, time_window, encoded_space_dim), dtype=torch.float32).to(device)
output_seq_pred = np.zeros((50, 50, encoded_space_dim))

# Loop through simulations
for sim_num in range(50):
    # Set up initial inputs
    input_seq[0, :, :] = torch.tensor(encoded_lstm_train[sim_num, :time_window, :], dtype=torch.float32).to(device)

    # Copy initial sequence to the output predictions
    output_seq_pred[sim_num, :time_window, :] = encoded_lstm_train[sim_num, :time_window, :]

    for t in range(time_window, 50):
        # Predict next time step
        with torch.no_grad():
            next_pred = lstm_model(input_seq).cpu().numpy()  # Forward pass through LSTM

        # Update output sequence predictions
        output_seq_pred[sim_num, t, :] = next_pred[0, :]

        # Update input sequence for next prediction
        input_seq[0, :-1, :] = input_seq[0, 1:, :].clone()  # Shift inputs left
        input_seq[0, -1, :] = torch.tensor(next_pred[0, :], dtype=torch.float32).to(device)  # Append new prediction

output_seq_pred = output_seq_pred  * std_encoded + mean_encoded

encoded, decoded, decoded_unnorm, data_unnorm, loss = ae._decode(output_seq_pred.reshape(50*50, encoded_space_dim), torch.tensor(lstm_train_data_norm.reshape(50*50, 3, 32, 32, 32), dtype=torch.float32, device=device))

print('after lstm recon error ', np.mean((decoded_unnorm-data_unnorm)**2))
np.save(f'./loss_after_lstm_train_{method}_{encoded_space_dim}.npy', loss)
# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
vmin = data_unnorm[1848, 1, :, 29, :].min()
vmax = data_unnorm[1848, 1, :, 29, :].max()
# First subplot: Sine Wave
axes[0].imshow(data_unnorm[1848, 1, :, 29, :], cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title("GT")

# Second subplot: Cosine Wave
axes[1].imshow(decoded_unnorm[1848, 1, :, 29, :], cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title("Model Prediction")


# Adjust layout
plt.tight_layout()

# Show the plot
plt.savefig(f'./comparison_train_after_lstm_{method}_{encoded_space_dim}.png')
plt.show()

print(loss)

