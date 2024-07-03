import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse

torch.set_printoptions(precision=8)
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# Set seed for reproducibility
torch.manual_seed(1990)
np.random.seed(1990)

print("GPU available?", torch.cuda.is_available())


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder, self).__init__()
        self.enc1 = nn.Conv3d(3, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(128)
        self.enc2 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(256)
        self.enc3 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(512)
        self.enc4 = nn.Conv3d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(1024)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024 * 2 * 2 * 2, encoded_space_dim)

    def forward(self, x):
        x = nn.SiLU()(self.bn1(self.enc1(x)))
        x = nn.SiLU()(self.bn2(self.enc2(x)))
        x = nn.SiLU()(self.bn3(self.enc3(x)))
        x = nn.SiLU()(self.bn4(self.enc4(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(encoded_space_dim, 1024 * 2 * 2 * 2)
        self.unflatten = nn.Unflatten(1, (1024, 2, 2, 2))
        self.dec1 = nn.ConvTranspose3d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm3d(512)
        self.dec2 = nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm3d(256)
        self.dec3 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.dec4 = nn.ConvTranspose3d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = nn.SiLU()(self.bn1(self.dec1(x)))
        x = nn.SiLU()(self.bn2(self.dec2(x)))
        x = nn.SiLU()(self.bn3(self.dec3(x)))
        x = self.dec4(x)
        return x


class Autoencoder:
    def __init__(self, rank, method, network_structure=None, normalize=True):
        self.method = method
        self.r = rank
        self.normalize = normalize
        self.network_structure = network_structure  # only showing half of the NN
        assert self.network_structure[-1] == self.r, "check your network structure"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'mps'

    def fit(self, X, lr, batch_size, epochs, X_test, save_path):
        """X: shape = (n_samples, n_dof)"""
        assert X.dtype == np.float32
        X_data = X.copy()
        X_data = torch.from_numpy(X_data).float().to(self.device)
        if self.normalize:
            self.mx = torch.max(X_data, axis=0).values
            self.sx = torch.min(X_data, axis=0).values
            X_data = self._normalize(X_data)
        dataset = TensorDataset(X_data)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        X_data_pod = X_data.reshape(X_data.shape[0], -1)

        if self.method == 'pod':
            self.encoder, self.decoder = self._get_svd(X_data_pod)

        elif self.method == 'ae':
            encoder_mlp = Encoder(self.r).to(self.device)
            decoder_mlp = Decoder(self.r).to(self.device)
            if torch.cuda.device_count() > 1:
                encoder_mlp = nn.DataParallel(encoder_mlp)
                decoder_mlp = nn.DataParallel(decoder_mlp)
            self.encoder = lambda x: encoder_mlp(x)
            self.decoder = lambda x: decoder_mlp(x)
            self.ParameterList = [*encoder_mlp.parameters(), *decoder_mlp.parameters()]

        elif self.method == 'pod-ae':
            encoder_pod, decoder_pod = self._get_svd(X_data_pod)
            encoder_mlp = Encoder(self.r).to(self.device)
            decoder_mlp = Decoder(self.r).to(self.device)
            if torch.cuda.device_count() > 1:
                encoder_mlp = nn.DataParallel(encoder_mlp)
                decoder_mlp = nn.DataParallel(decoder_mlp)
            self.encoder = lambda x: encoder_pod(x.reshape(x.shape[0], -1)) + encoder_mlp(x)
            self.decoder = lambda x: decoder_pod(x).reshape(x.shape[0], *X_data.shape[1:]) + decoder_mlp(x)
            self.ParameterList = [*encoder_mlp.parameters(), *decoder_mlp.parameters()]

        elif self.method == 'pod-ae-tunable':
            encoder_pod, decoder_pod = self._get_svd(X_data_pod)
            encoder_mlp = Encoder(self.r).to(self.device)
            decoder_mlp = Decoder(self.r).to(self.device)
            if torch.cuda.device_count() > 1:
                encoder_mlp = nn.DataParallel(encoder_mlp)
                decoder_mlp = nn.DataParallel(decoder_mlp)
            self.a = nn.Parameter(0.5 * torch.zeros(1).to(self.device))
            self.b = nn.Parameter(0.5 * torch.zeros(1).to(self.device))
            self.encoder = lambda x: (1.0 - self.a) * encoder_pod(x.reshape(x.shape[0], -1)) + self.a * encoder_mlp(x)
            self.decoder = lambda x: (1.0 - self.b) * decoder_pod(x).reshape(x.shape[0],
                                                                             *X_data.shape[1:]) + self.b * decoder_mlp(
                x)
            self.ParameterList = [*encoder_mlp.parameters(), *decoder_mlp.parameters()]

        self.loss_fn = nn.MSELoss()
        plt_loss = []
        plt_loss_test = []
        if self.method == 'pod':
            with torch.no_grad():
                data_tensor = dataset.tensors[0].to(self.device)
                total_loss = self.loss_fn(
                    self.decoder(self.encoder(data_tensor.reshape(data_tensor.shape[0], -1))).reshape(
                        data_tensor.shape[0], *X_data.shape[1:]), data_tensor)
                X_data_test = self._normalize(torch.from_numpy(X_test).float().to(self.device))
                decoded_test = self.decoder(self.encoder(X_data_test.reshape(X_data_test.shape[0], -1))).reshape(
                    X_data_test.shape[0], *X_data_test.shape[1:])
                total_loss_test = self.loss_fn(decoded_test, X_data_test)
                # self._visualize_reconstruction(decoded_test, X_data_test, 0, save_path)
                self._reporting_loss(self.decoder(self.encoder(data_tensor.reshape(data_tensor.shape[0], -1))).reshape(
                        data_tensor.shape[0], *X_data.shape[1:]), data_tensor, decoded_test, X_data_test, grid, 0)
                print(f"POD total dataset loss = {total_loss:.5e}")
                print(f"POD test loss = {total_loss_test:.5e}")

        else:
            if self.method == 'pod-ae-tunable':
                self.optimizer = optim.Adam([{'params': self.ParameterList, 'lr': lr},
                                             {'params': [self.a, self.b], 'lr': 0.1 * lr}])

            else:
                self.optimizer = optim.Adam([
                    {'params': self.ParameterList, 'lr': lr}
                ])
            scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, verbose=True, min_lr=1e-5,
                                                       patience=21)
            # scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6, verbose=False)
            loss_vs_epoch_train = []
            loss_vs_epoch_test = []
            epochs_list = []
            warmup_epochs = 1000
            for epoch in range(epochs):
                for batch_data in dataloader:
                    batch_x = batch_data[0].to(self.device)
                    self.optimizer.zero_grad()
                    output = self.decoder(self.encoder(batch_x))
                    loss = self.loss_fn(output, batch_x)
                    loss.backward()
                    self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.ParameterList, max_norm=1.0)
                    if method=='pod-ae-tunable':
                        torch.nn.utils.clip_grad_norm_(self.a, max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.b, max_norm=1.0)

                scheduler.step(loss)

                if (epoch) % 10 == 0:
                    with torch.no_grad():
                        total_loss = self.loss_fn(self.decoder(self.encoder(dataset.tensors[0].to(self.device))),
                                                  dataset.tensors[0].to(self.device))
                        X_data_test = self._normalize(torch.from_numpy(X_test).float().to(self.device))
                        decoded_test = self.decoder(self.encoder(X_data_test))
                        total_loss_test = self.loss_fn(decoded_test, X_data_test)
                        print(f'Epoch: {epoch}, Total train Loss: {total_loss:.5e}')
                        print(f'Epoch: {epoch}, Total test Loss: {total_loss_test:.5e}')
                        if method=='pod-ae-tunable':
                            print(f'a = {self.a.detach().cpu()[0]:.5e}, b={self.b.detach().cpu()[0]:.5e}')
                        # self._visualize_reconstruction(decoded_test, X_data_test, epoch, save_path)
                        plt_loss.append(total_loss.item())
                        plt_loss_test.append(total_loss_test.item())
                        epochs_list.append(epoch)
                        self._reporting_loss(self.decoder(self.encoder(dataset.tensors[0].to(self.device))), dataset.tensors[0].to(self.device),
                                             decoded_test, X_data_test, grid, epoch)
                        # np.save(save_path + '/train_loss', plt_loss)
                        # np.save(save_path + '/test_loss', plt_loss_test)
                        # np.save(save_path + '/epoch_list', epochs_list)

    def _reporting_loss(self, decoded_train, train, decoded_test, test, grid, epoch):

        decoded_train = self._unnormalize(decoded_train)
        train = self._unnormalize(train)
        decoded_test = self._unnormalize(decoded_test)
        test = self._unnormalize(test)

        report_train_loss = torch.sqrt((1 / decoded_train.shape[0]) *
                                      torch.sum((decoded_train - train) ** 2) * (2 * torch.pi / grid) ** 3)

        report_test_loss =  torch.sqrt((1 / decoded_test.shape[0])*
            torch.sum((decoded_test - test) ** 2) * (2 * torch.pi / grid) ** 3)

        print(f'Epoch: {epoch}, reporting train Loss: {report_train_loss:.5e}')
        print(f'Epoch: {epoch}, reporting test Loss: {report_test_loss:.5e}')

        if self.method == 'pod-ae-tunable':
            n_parameters_mlp = sum(p.numel() for p in self.ParameterList if p.requires_grad)
            n_parameters_svd = self.vhr.numel()
            n_parameters_others = 2
        elif self.method=='ae':
            n_parameters_mlp = sum(p.numel() for p in self.ParameterList if p.requires_grad)
            n_parameters_svd = -99
            n_parameters_others = -99
        elif self.method=='pod':
            n_parameters_mlp = -99
            n_parameters_svd = self.vhr.numel()
            n_parameters_others = -99
        else:
            n_parameters_mlp = -99
            n_parameters_svd = -99
            n_parameters_others = -99
        np.savez(save_path+'/quantities_of_interest.npz', decoded_train=decoded_train.detach().cpu().numpy(), train=train.detach().cpu().numpy(),
                 decoded_test=decoded_test.detach().cpu().numpy(), test=test.detach().cpu().numpy(),
                 n_parameters_mlp=n_parameters_mlp, n_parameters_svd=n_parameters_svd, n_parameters_others=n_parameters_others,
                 report_train_loss = report_train_loss.detach().cpu().numpy(), report_test_loss = report_test_loss.detach().cpu().numpy())


        return None

    def _save_checkpoint(self, epoch, save_path, encoder_mlp, decoder_mlp):
        encoder_path = os.path.join(save_path, f'encoder_epoch_{epoch}.pth')
        decoder_path = os.path.join(save_path, f'decoder_epoch_{epoch}.pth')
        a_path = os.path.join(save_path, f'a_epoch_{epoch}.pth')
        b_path = os.path.join(save_path, f'b_epoch_{epoch}.pth')
        torch.save(encoder_mlp.state_dict(), encoder_path)
        torch.save(decoder_mlp.state_dict(), decoder_path)
        if self.method == 'pod-ae-tunable':
            torch.save(self.a, a_path)
            torch.save(self.b, b_path)
        print(f"Model checkpoints saved at epoch {epoch}")

    def _visualize_reconstruction(self, decoded_test, X_test_data, epoch, save_path):

        decoded_test_unnormalized = self._unnormalize(decoded_test).reshape(
            (decoded_test.shape[0], 3, grid, grid, grid))
        X_test_data_unnormalized = self._unnormalize(X_test_data).reshape(
            (decoded_test.shape[0], 3, grid, grid, grid))
        error_squared = (decoded_test_unnormalized - X_test_data_unnormalized) ** 2.0

        # Generate tick positions and labels for every 10 points
        tick_positions = np.arange(0, X_test_data_unnormalized.shape[2], 10)
        tick_labels = np.linspace(0, 2 * np.pi, X_test_data_unnormalized.shape[2])[::10]

        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        font = {'family': 'serif', 'weight': 'normal', 'size': 14}
        plt.rc('font', **font)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        for i in range(3):

            # Display the image
            im = axs[i, 0].imshow(decoded_test_unnormalized[0, i, 0, :, :].detach().cpu().numpy(), cmap='viridis')

            # Customize x-axis ticks
            axs[i, 0].set_xticks(tick_positions)
            axs[i, 0].set_xticklabels([f'{label:.2f}' for label in tick_labels], fontsize=14, family='serif',
                                      rotation=90)

            # Customize y-axis ticks and flip the labels
            axs[i, 0].set_yticks(tick_positions)
            axs[i, 0].set_yticklabels([f'{label:.2f}' for label in tick_labels[::-1]], fontsize=14, family='serif')

            # Add labels and title for clarity
            axs[i, 0].set_xlabel('Y', fontsize=14)
            axs[i, 0].set_ylabel('Z', fontsize=14)
            cbar = fig.colorbar(im, ax=axs[i, 0], orientation='vertical')
            cbar.ax.tick_params(labelsize=12)

            if i == 0:
                axs[i, 0].set_title(f'Re-construction (u)', fontsize=16, family='serif')
            if i == 1:
                axs[i, 0].set_title(f'Re-construction (v)', fontsize=16, family='serif')
            if i == 2:
                axs[i, 0].set_title(f'Re-construction (w)', fontsize=16, family='serif')

            im = axs[i, 1].imshow(X_test_data_unnormalized[0, i, 0, :, :].detach().cpu().numpy(), cmap='viridis')

            # Customize x-axis ticks
            axs[i, 1].set_xticks(tick_positions)
            axs[i, 1].set_xticklabels([f'{label:.2f}' for label in tick_labels], fontsize=14, family='serif',
                                      rotation=90)

            # Customize y-axis ticks and flip the labels
            axs[i, 1].set_yticks(tick_positions)
            axs[i, 1].set_yticklabels([f'{label:.2f}' for label in tick_labels[::-1]], fontsize=14, family='serif')

            # Add labels and title for clarity
            axs[i, 1].set_xlabel('Y', fontsize=14)
            axs[i, 1].set_ylabel('Z', fontsize=14)
            cbar = fig.colorbar(im, ax=axs[i, 1], orientation='vertical')
            cbar.ax.tick_params(labelsize=12)

            if i == 0:
                axs[i, 1].set_title(f'Ground Truth (u)', fontsize=16, family='serif')
            if i == 1:
                axs[i, 1].set_title(f'Ground Truth (v)', fontsize=16, family='serif')
            if i == 2:
                axs[i, 1].set_title(f'Ground Truth (w)', fontsize=16, family='serif')

            im = axs[i, 2].imshow(error_squared[0, i, 0, :, :].detach().cpu().numpy(), cmap='viridis')

            # Customize x-axis ticks
            axs[i, 2].set_xticks(tick_positions)
            axs[i, 2].set_xticklabels([f'{label:.2f}' for label in tick_labels], fontsize=14, family='serif',
                                      rotation=90)

            # Customize y-axis ticks and flip the labels
            axs[i, 2].set_yticks(tick_positions)
            axs[i, 2].set_yticklabels([f'{label:.2f}' for label in tick_labels[::-1]], fontsize=14, family='serif')

            # Add labels and title for clarity
            axs[i, 2].set_xlabel('Y', fontsize=14)
            axs[i, 2].set_ylabel('Z', fontsize=14)
            fig.colorbar(im, ax=axs[i, 2], orientation='vertical')
            cbar.ax.tick_params(labelsize=12)

            if i == 0:
                axs[i, 2].set_title(f'Squared Error (u)', fontsize=16, family='serif')
            if i == 1:
                axs[i, 2].set_title(f'Squared Error (v)', fontsize=16, family='serif')
            if i == 2:
                axs[i, 2].set_title(f'Squared Error (w)', fontsize=16, family='serif')

        # Show the plot
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_path + '/visual_' + str(epoch) + '.png')
        return None

    def _get_svd(self, X):
        X_cpu = np.float32(input_data.reshape(input_data.shape[0], -1))
        # X_cpu = X.detach().cpu().numpy()
        u, s, vh = np.linalg.svd(X_cpu, full_matrices=False)

        s = torch.from_numpy(s).to(self.device)
        vh = torch.from_numpy(vh).to(self.device)

        # u,s,vh = torch.linalg.svd(X,full_matrices=False)
        # ur = u[:,:self.r]
        # sr = sr[:self.r]
        self.vhr = vh[:self.r, :]
        encoder = lambda x: torch.matmul(x, self.vhr.H)
        decoder = lambda h: torch.matmul(h, self.vhr)
        return encoder, decoder

    def _normalize(self, X):
        X = (X - self.sx) / (self.mx - self.sx)
        return X

    def _unnormalize(self, X):
        X = X * (self.mx - self.sx) + self.sx
        return X

    def predict_hidden(self, X):
        with torch.no_grad():
            X_data = X.copy()
            assert X.dtype == np.float32
            X_data = torch.from_numpy(X_data).float().to(self.device)
            if self.normalize:
                X_data = self._normalize(X_data)
            y = self.encoder(X_data)
            y = y.detach().cpu().numpy()
        return y

    def predict(self, X):
        with torch.no_grad():
            X_data = X.copy()
            X_data = torch.from_numpy(X_data).to(self.device)
            if self.normalize:
                X_data = self._normalize(X_data)
            y = self.encoder(X_data)
            x_rec = self.decoder(y)
            if self.normalize:
                x_rec = self._unnormalize(x_rec)
            x_rec = x_rec.detach().cpu().numpy()
        return x_rec


# Initialize argument parser
parser = argparse.ArgumentParser(description='Autoencoder Arguments')

# Add arguments
parser.add_argument('--grid', type=int, default=128, help='Grid dimension')
parser.add_argument('--encoded_space_dim', type=int, default=15, help='Dimension of the encoded space')
parser.add_argument('--method', type=str, default='ae', choices=['ae', 'pod', 'pod-ae', 'pod-ae-tunable'],
                    help='Method for autoencoder')

# Parse arguments
args = parser.parse_args()

# Access arguments
grid = args.grid
encoded_space_dim = args.encoded_space_dim
method = args.method

rank = encoded_space_dim

save_path = './hit_' + method + '_' + str(grid) + '_' + str(encoded_space_dim)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Folder '{save_path}' created.")
else:
    print(f"Folder '{save_path}' already exists.")

input_data = np.zeros((100, 3, grid, grid, grid))
for tstep in range(100):
    data = np.load(f'./U_{grid}_recheck/{tstep}.npz')
    u_reshaped = data['u']
    v_reshaped = data['v']
    w_reshaped = data['w']
    input_data[tstep, 0, :, :, :] = u_reshaped
    input_data[tstep, 1, :, :, :] = v_reshaped
    input_data[tstep, 2, :, :, :] = w_reshaped

X = np.float32(input_data)
# np.random.shuffle(X)
# Split the data into training and test sets
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Define split size
split_index = int(0.7 * num_samples)

# Split indices
train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Split the data
X_train = X[train_indices]
X_test = X[test_indices]

network_structure = [X.shape[1], rank * 100, rank]
batch_size = 20
normalize = True
epochs = 1000
learning_rate = 1e-3  # 0.001

ae = Autoencoder(rank, method, network_structure, normalize)
ae.fit(X_train, learning_rate, batch_size, epochs, X_test, save_path)
