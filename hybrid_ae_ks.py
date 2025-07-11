# idea: POD combined with ANN for an autoencoder that guarantee precision
# ANN: x -> g_enc(x) -> g_dec(g_enc(x))
# POD: x -> U^Tx -> UU^T x
# 
# Motivation: from PirateNet idea, we know a good initial guess matters a lot
# Thus, we consider 
# option 1: 
# xhat = a*g_dec(g_enc(x)) + (1-a)*Uy 
#   latent space: g_enc(x) + U^T x
#   - -: this leads to unclear latent space
#   - +: it might be good enough for reconstruction, but who cares? 
# option 2: 
# y = a*g_enc(x) + (1-a)*U^T x 
#   -> y = a*mlp_encoder(x) + (1-a)*pod_encoder(x)
# xhat = b*g_dec(y) + (1-b)*U*y 
#   -> xhat = b*mlp_decoder(y) + (1-b)*pod_decoder(y) 
# 
# note: 
# - when a = 0, b=0, this is POD
# - when a = 1, b=1, this is MLP 
# 
# we initialize "trainable" parameters a = 0 and b = 0
# let's see how it works compared to vanilla autoencoder and my hybrid POD idea.

# optionally: think about layerwise a,b

import numpy as np
from scipy.io import loadmat

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
torch.set_printoptions(precision=8)
import argparse
import torch.optim.lr_scheduler as lr_scheduler
# Set seed for reproducibility
torch.manual_seed(1990)
np.random.seed(1990)

print("GPU available?", torch.cuda.is_available())

class MLP(nn.Module):
    def __init__(self, network_structure):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(network_structure) - 1):
            self.layers.append(nn.Linear(network_structure[i], network_structure[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        x = self.layers[-1](x)
        return x


class Autoencoder:
    def __init__(self, rank, method, network_structure=None, normalize=True, device='cuda:0'):
        self.method = method 
        self.r = rank
        self.normalize = normalize
        self.network_structure = network_structure # only showing half of the NN
        assert self.network_structure[-1] == self.r, "check your network structure"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = 'mps'
        self.device = device

    def train(self, X, lr, batch_size, epochs, X_test, save_path):
        """X: shape = (n_samples, n_dof)"""
        assert X.dtype == np.float32
        X_data = X.copy()
        X_data = torch.from_numpy(X_data).float().to(self.device)
        if self.normalize:
            self.mx = torch.mean(X_data,axis=0)
            self.sx = torch.std(X_data,axis=0)
            X_data = self._normalize(X_data)
        dataset = TensorDataset(X_data)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        if self.method == 'pod':
            self.encoder, self.decoder = self._get_svd(X_data)

        elif self.method == 'ae':
            encoder_mlp = MLP(network_structure=self.network_structure).to(self.device)
            deocder_mlp = MLP(network_structure=self.network_structure[::-1]).to(self.device)
            self.encoder = lambda x: encoder_mlp(x)
            self.decoder = lambda x: deocder_mlp(x)
            self.ParameterList = [*encoder_mlp.parameters(), *deocder_mlp.parameters()]

        elif self.method == 'pod-ae':
            encoder_pod, decoder_pod = self._get_svd(X_data)
            encoder_mlp = MLP(network_structure=self.network_structure).to(self.device)
            deocder_mlp = MLP(network_structure=self.network_structure[::-1]).to(self.device)
            self.encoder = lambda x: encoder_pod(x) + encoder_mlp(x)
            self.decoder = lambda x: decoder_pod(x) + deocder_mlp(x)
            self.ParameterList = [*encoder_mlp.parameters(), *deocder_mlp.parameters()]

        elif self.method == 'pod-ae-tunable':
            encoder_pod, decoder_pod = self._get_svd(X_data)
            encoder_mlp = MLP(network_structure=self.network_structure).to(self.device)
            deocder_mlp = MLP(network_structure=self.network_structure[::-1]).to(self.device)
            self.a = nn.Parameter(0.1*torch.zeros(1).to(self.device))
            self.b = nn.Parameter(0.1*torch.zeros(1).to(self.device))
            self.encoder = lambda x: (1.0-self.a)*encoder_pod(x) + (self.a)*encoder_mlp(x)
            self.decoder = lambda x: (1.0-self.b)*decoder_pod(x) + (self.b)*deocder_mlp(x)
            self.ParameterList = [*encoder_mlp.parameters(), *deocder_mlp.parameters()]

        elif self.method == 'naive':
            encoder_pod, decoder_pod = self._get_svd(X_data)
            encoder_mlp = MLP(network_structure=self.network_structure).to(self.device)
            deocder_mlp = MLP(network_structure=self.network_structure[::-1]).to(self.device)
            self.encoder = lambda x: encoder_pod(x) + encoder_mlp(x)
            self.decoder = lambda x: decoder_pod(x) + deocder_mlp(x)
            self.ParameterList = [*encoder_mlp.parameters(), *deocder_mlp.parameters()]

        self.loss_fn = nn.MSELoss()
        plt_loss = []
        plt_loss_test = []
        if self.method == 'pod':
            with torch.no_grad():
                total_loss = self.loss_fn(self.decoder(self.encoder(dataset.tensors[0].to(self.device))), dataset.tensors[0].to(self.device))
                X_data_test = self._normalize(torch.from_numpy(X_test).float().to(self.device))
                total_loss_test = self.loss_fn(self.decoder(self.encoder(X_data_test)), X_data_test)
                print(f"POD total dataset loss = {total_loss:.5e}")
                print(f'Total test Loss: {total_loss_test:.5e}')
                plt_loss.append(total_loss.item())
                plt_loss_test.append(total_loss_test.item())
                test_decoded_unnorm = self._unnormalize(self.decoder(self.encoder(X_data_test)))

                # np.save(save_path + '/train_loss', total_loss.item())
                # np.save(save_path + '/test_loss', total_loss_test.item())
                # np.save(save_path + '/epoch_list', 0)
                # np.save(save_path+'/vhr', self.vhr.cpu().numpy())
        else:
            if self.method == 'pod-ae-tunable':
                self.optimizer = optim.Adam([
                    {'params': [self.a, self.b], 'lr': lr * 0.1},
                    {'params': self.ParameterList, 'lr': lr}
                ])
            else:
                self.optimizer = optim.Adam([
                    {'params': self.ParameterList, 'lr': lr}
                ])
            loss_vs_epoch_train = []
            loss_vs_epoch_test = []
            epochs_list = []
            # scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, verbose=True, min_lr=1e-6,
            #                                            patience=100)
            a_list = []
            b_list=[]
            for epoch in range(epochs):
                for batch_data in dataloader:
                    batch_x = batch_data[0].to(self.device)
                    self.optimizer.zero_grad()
                    output = self.decoder(self.encoder(batch_x))
                    loss = self.loss_fn(output, batch_x)
                    loss.backward()
                    self.optimizer.step()
                # scheduler.step(loss)
                if (epoch) % 100 == 0:
                    with torch.no_grad():
                        total_loss = self.loss_fn(self.decoder(self.encoder(dataset.tensors[0].to(self.device))), dataset.tensors[0].to(self.device))
                        X_data_test = self._normalize(torch.from_numpy(X_test).float().to(self.device))
                        total_loss_test = self.loss_fn(self.decoder(self.encoder(X_data_test)), X_data_test)
                        print(f'Epoch: {epoch}, Total train Loss: {total_loss:.5e}')
                        print(f'Epoch: {epoch}, Total test Loss: {total_loss_test:.5e}')
                        if self.method == 'pod-ae-tunable':
                            print(f'a = {self.a.detach().cpu()[0]:.5e}, b={self.b.detach().cpu()[0]:.5e}')
                            a_list.append(self.a.detach().cpu()[0])
                            b_list.append(self.b.detach().cpu()[0])
                        plt_loss.append(total_loss.detach().cpu().numpy())
                        plt_loss_test.append(total_loss_test.detach().cpu().numpy())
                        test_decoded_unnorm = self._unnormalize(self.decoder(self.encoder(X_data_test)))
                        # plt.figure(figsize=(4,4))
                        # plt.semilogy(plt_loss, label='train')
                        # plt.semilogy(plt_loss_test, label='test')
                        # plt.legend()
                        # plt.savefig(save_path + "/loss_{self.method}.png",dpi=150)
                        # plt.close()
                        # if self.method=='pod-ae-tunable':
                        #     checkpoint = {
                        #         'ParameterList': self.ParameterList,
                        #         'vhr': self.vhr
                        #     }
                        # elif self.method=='ae':
                        #     checkpoint = {
                        #         'ParameterList': self.ParameterList
                        #     }
                        # torch.save(checkpoint, save_path + '/epoch_'+str(epoch)+'.pth')
                        # loss_vs_epoch_train.append(total_loss.item())
                        # loss_vs_epoch_test.append(total_loss_test.item())
                        # epochs_list.append(epoch)
            np.save(save_path + '/train_loss', np.array(plt_loss))
            np.save(save_path + '/test_loss', np.array(plt_loss_test))
            if self.method=='pod-ae-tunable':
                np.save(save_path + '/a_list', np.array(a_list))
                np.save(save_path + '/b_list', np.array(b_list))
            np.save(save_path + '/test_data', X_test)
            np.save(save_path + '/test_data_decoded', test_decoded_unnorm.detach().cpu().numpy())
        return None

    def _get_svd(self, X):

        X_cpu = X.detach().cpu().numpy()
        u,s,vh = np.linalg.svd(X_cpu, full_matrices=False)

        s = torch.from_numpy(s).to(self.device)
        vh = torch.from_numpy(vh).to(self.device)

        # u,s,vh = torch.linalg.svd(X,full_matrices=False)
        # ur = u[:,:self.r]
        # sr = sr[:self.r]
        self.vhr = vh[:self.r,:]
        encoder = lambda x: torch.matmul(x, self.vhr.H)
        decoder = lambda h: torch.matmul(h, self.vhr)
        return encoder, decoder

    def _normalize(self, X):
        X = (X - self.mx) / self.sx
        return X

    def _unnormalize(self, X):
        X = X*self.sx + self.mx
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


parser = argparse.ArgumentParser(description='Autoencoder Arguments')

# Add arguments
parser.add_argument('--grid', type=int, default=128, help='Grid dimension')
parser.add_argument('--encoded_space_dim', type=int, default=15, help='Dimension of the encoded space')
parser.add_argument('--method', type=str, default='ae', choices=['ae', 'pod', 'pod-ae', 'pod-ae-tunable', 'naive'], help='Method for autoencoder')

# Parse arguments
args = parser.parse_args()

# Access arguments
grid = args.grid
encoded_space_dim = args.encoded_space_dim
method = args.method

X = loadmat(f'ks_{grid}_extended.mat')['data']
X = np.float32(X).T
print(f"data shape = {X.shape}")

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

rank = encoded_space_dim
#method = 'pod'
#method = 'ae'
#method = 'pod-ae'
#method = 'pod-ae-tunable'
#grid = 2048
#encoded_space_dim = rank
save_path = './ks_' + method + '_' + str(grid) + '_' + str(encoded_space_dim)

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Folder '{save_path}' created.")
else:
    print(f"Folder '{save_path}' already exists.")

network_structure = [grid, rank*2, rank]
batch_size = 32
normalize = True
epochs = 40000
learning_rate = 1e-4 # 0.001
device = 'cuda'

ae = Autoencoder(rank, method, network_structure, normalize, device)
ae.train(X_train, learning_rate, batch_size, epochs, X_test, save_path)

