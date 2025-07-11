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
parser.add_argument('--grid', type=int, default=128, help='Grid dimension')
parser.add_argument('--encoded_space_dim', type=int, default=15, help='Dimension of the encoded space')
parser.add_argument('--method', type=str, default='ae', choices=['ae', 'pod', 'pod-ae', 'pod-ae-tunable', 'naive'],
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
        
        index_rand = np.arange(X.shape[0])
        np.random.shuffle(blah)
        X = X[index_rand]
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        vhr = torch.from_numpy(vh[:self.r, :]).to(self.device)

        return vhr

    def _init_encoder_decoder(self, X):
        
        if self.method == 'pod':
            self.vhr = self._get_svd(X_train_unnorm.reshape(X_train_unnorm.shape[0], -1))
            self.encoder = lambda x: torch.matmul(x, self.vhr.T)
            self.decoder = lambda x: torch.matmul(x, self.vhr)

        elif self.method == 'ae':
            self.encoder_mlp = Encoder(self.r).to(self.device)
            self.decoder_mlp = Decoder(self.r).to(self.device)
            if torch.cuda.device_count() > 1:
                self.encoder_mlp = nn.DataParallel(self.encoder_mlp)
                self.decoder_mlp = nn.DataParallel(self.decoder_mlp)
            #self.encoder = lambda x: encoder_mlp(x)
            #self.decoder = lambda x: decoder_mlp(x)
            self.ParameterList = [*self.encoder_mlp.parameters(), *self.decoder_mlp.parameters()]

        elif self.method == 'pod-ae-tunable':
            self.encoder_mlp = Encoder(self.r).to(self.device)
            self.decoder_mlp = Decoder(self.r).to(self.device)
            self.vhr = self._get_svd(X_train_unnorm.reshape(X_train_unnorm.shape[0], -1))
            self.encoder_pod = lambda x: torch.matmul(x, self.vhr.T)
            self.decoder_pod = lambda x: torch.matmul(x, self.vhr)
            if torch.cuda.device_count() > 1:
                self.encoder_mlp = nn.DataParallel(self.encoder_mlp)
                self.decoder_mlp = nn.DataParallel(self.decoder_mlp)
            self.a = nn.Parameter(0.01 * torch.zeros(self.r).to(self.device))
            self.b = nn.Parameter(0.05 * torch.zeros(3).to(self.device))
            self.ParameterList = [*self.encoder_mlp.parameters(), *self.decoder_mlp.parameters()]

        elif self.method == 'naive':
            self.encoder_mlp = Encoder(self.r).to(self.device)
            self.decoder_mlp = Decoder(self.r).to(self.device)
            self.vhr = self._get_svd(X_train_unnorm.reshape(X_train_unnorm.shape[0], -1))
            self.encoder_pod = lambda x: torch.matmul(x, self.vhr.T)
            self.decoder_pod = lambda x: torch.matmul(x, self.vhr)
            if torch.cuda.device_count() > 1:
                self.encoder_mlp = nn.DataParallel(self.encoder_mlp)
                self.decoder_mlp = nn.DataParallel(self.decoder_mlp)
            self.ParameterList = [*self.encoder_mlp.parameters(), *self.decoder_mlp.parameters()]
        else:
            print('Unknown Method')

    def _fit(self, X_train, X_test, batch_size, epochs, lr):
        dataset = TensorDataset(torch.from_numpy(X_train).float().to(self.device))
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        self.loss_fn = nn.MSELoss()
        plt_loss = []
        plt_loss_test = []
        epoch_list = []
        a_list = []
        b_list = []

        if self.method == 'pod':
            with torch.no_grad():
                train_dataset = dataset.tensors[0].to(self.device)
                encoded_train = self.encoder(train_dataset.reshape(train_dataset.shape[0], -1))
                decoded_train = self.decoder(encoded_train).reshape(train_dataset.shape[0], *train_dataset.shape[1:])
                train_loss = self.loss_fn(train_dataset, decoded_train)

                test_dataset = torch.from_numpy(X_test).to(self.device)
                encoded_test = self.encoder(test_dataset.reshape(test_dataset.shape[0], -1))
                decoded_test = self.decoder(encoded_test).reshape(test_dataset.shape[0], *train_dataset.shape[1:])
                test_loss = self.loss_fn(test_dataset, decoded_test)

                decoded_test_unnorm, test_dataset_unnorm = self._visualize_reconstruction(decoded_test, test_dataset, 0)

                decoded_train_unnorm = decoded_train.detach().cpu().numpy() * std_value + mean_value
                train_unnorm = train_dataset.detach().cpu().numpy() * std_value + mean_value

                plt_loss.append(train_loss.item())
                plt_loss_test.append(test_loss.item())
                print(f"POD total dataset loss = {train_loss:.5e}")
                print(f"POD test loss = {test_loss:.5e}")

                np.save(save_path + '/train_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        np.array(plt_loss))
                np.save(save_path + '/test_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        np.array(plt_loss_test))
                np.save(save_path + '/epochs_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        np.array(epoch_list))
                np.save(save_path + '/a_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        np.array(a_list))
                np.save(save_path + '/b_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        np.array(b_list))
                np.save(save_path + '/decoded_test_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        decoded_test_unnorm)
                np.save(save_path + '/test_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        test_dataset_unnorm)
                np.save(
                    save_path + '/decoded_train_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                    decoded_train_unnorm)
                np.save(save_path + '/train_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                        train_unnorm)

        if self.method == 'ae':
            self.optimizer = torch.optim.Adam([
                {'params': self.ParameterList, 'lr': lr}
            ])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, min_lr=1e-6,
                                                                   patience=100)
            for epoch in range(epochs):
                loss_epoch = 0.0
                for batch_data in dataloader:
                    batch_data = batch_data[0].to(self.device)
                    self.optimizer.zero_grad()
                    output = self.decoder_mlp(self.encoder_mlp(batch_data))
                    loss = self.loss_fn(output, batch_data)
                    loss.backward()
                    self.optimizer.step()
                    loss_epoch += loss.detach().cpu().numpy()
                    # torch.nn.utils.clip_grad_norm_(self.ParameterList, max_norm=1.0)
                # scheduler.step(loss, epoch)
                loss_epoch = loss_epoch / len(dataloader)

                if (epoch) % 100 == 0:
                    self.encoder_mlp.eval()
                    self.decoder_mlp.eval()
                    with torch.no_grad():
                        # train_dataset = dataset.tensors[0].to(self.device)
                        # decoded_train = self.decoder(self.encoder(train_dataset))
                        total_loss = loss_epoch  # self.loss_fn(train_dataset, decoded_train)

                        test_dataset = torch.from_numpy(X_test).to(self.device)
                        # decoded_test = self.decoder(self.encoder(test_dataset))
                        split_size = test_dataset.shape[0] // 3  # Determine the split size
                        splits = torch.split(test_dataset,
                                             [split_size, split_size, test_dataset.shape[0] - 2 * split_size])
                        decoded_parts = []
                        for part in splits:
                            encoded_part = self.encoder_mlp(part)
                            decoded_part = self.decoder_mlp(encoded_part)
                            decoded_parts.append(decoded_part)

                        decoded_test = torch.cat(decoded_parts, dim=0)

                        train_dataset = torch.from_numpy(X_train).to(self.device)
                        # decoded_test = self.decoder(self.encoder(test_dataset))
                        split_size = train_dataset.shape[0] // 3  # Determine the split size
                        splits = torch.split(train_dataset,
                                             [split_size, split_size, train_dataset.shape[0] - 2 * split_size])

                        decoded_parts = []
                        for part in splits:
                            encoded_part = self.encoder_mlp(part)
                            decoded_part = self.decoder_mlp(encoded_part)
                            decoded_parts.append(decoded_part)

                        decoded_train = torch.cat(decoded_parts, dim=0)

                        decoded_train_unnorm = decoded_train.detach().cpu().numpy() * std_value + mean_value
                        train_unnorm = train_dataset.detach().cpu().numpy() * std_value + mean_value

                        total_loss_test = self.loss_fn(decoded_test, test_dataset)
                        total_loss_train = self.loss_fn(decoded_train, train_dataset)
                        print(f'Epoch: {epoch}, Total train Loss: {total_loss_train:.5e}')
                        print(f'Epoch: {epoch}, Total test Loss: {total_loss_test:.5e}')
                        decoded_test_unnorm, test_dataset_unnorm = self._visualize_reconstruction(decoded_test,
                                                                                                  test_dataset, epoch)
                        plt_loss.append(total_loss_train.item())
                        plt_loss_test.append(total_loss_test.item())
                        epoch_list.append(epoch)

                        np.save(
                            save_path + '/train_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            np.array(plt_loss))
                        np.save(save_path + '/test_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(plt_loss_test))
                        np.save(save_path + '/epochs_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(epoch_list))
                        np.save(save_path + '/a_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(a_list))
                        np.save(save_path + '/b_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(b_list))
                        np.save(
                            save_path + '/decoded_test_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            decoded_test_unnorm)
                        np.save(save_path + '/test_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                test_dataset_unnorm)
                        np.save(
                            save_path + '/decoded_train_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            decoded_train_unnorm)
                        np.save(
                            save_path + '/train_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            train_unnorm)
                        #torch.save({
                        #    'encoder_mlp_state_dict': self.encoder_mlp.state_dict(),
                        #    'decoder_mlp_state_dict': self.decoder_mlp.state_dict()
                        #}, os.path.join(save_path, f'{method}_parameters.pth'))

                    self.encoder_mlp.train()
                    self.decoder_mlp.train()
            a_list.append(-99)
            b_list.append(-99)
        if self.method == 'pod-ae-tunable':
            self.optimizer = torch.optim.Adam([
                {'params': self.ParameterList, 'lr': lr},
                {'params': [self.a, self.b], 'lr': lr*0.1}
            ])
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, min_lr=1e-5,
            #                                                        patience=100)
            for epoch in range(epochs):
                loss_epoch = 0.0
                for batch_data in dataloader:
                    batch_data = batch_data[0].to(self.device)
                    self.optimizer.zero_grad()
                    encoded_data = (1.0 - self.a) * self.encoder_pod(
                        batch_data.reshape(batch_data.shape[0], -1)) + self.a * self.encoder_mlp(batch_data)
                    decoded_data = (1.0 - self.b.view(1, 3, 1, 1, 1)) * self.decoder_pod(encoded_data).reshape(
                        encoded_data.shape[0],
                        *batch_data.shape[
                         1:]) + self.b.view(1, 3, 1, 1, 1) * self.decoder_mlp(
                        encoded_data)
                    loss = self.loss_fn(decoded_data, batch_data)
                    loss.backward()
                    self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.ParameterList, max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.a, max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.b, max_norm=1.0)
                    loss_epoch += loss.detach().cpu().numpy()
                # scheduler.step(loss, epoch)

                loss_epoch = loss_epoch / len(dataloader)

                if (epoch) % 100 == 0:
                    with torch.no_grad():
                        total_loss, total_loss_test, decoded_test_unnorm, test_dataset_unnorm = self._print_stats(
                            train_dataset=None, test_dataset=torch.from_numpy(X_test).to(self.device), epoch=epoch,
                            loss=loss_epoch)

                        epoch_list.append(epoch)
                        a_list.append(torch.norm(self.a).detach().cpu().numpy())
                        b_list.append(torch.norm(self.b).detach().cpu().numpy())


                        train_dataset = torch.from_numpy(X_train).to(self.device)
                        split_size = train_dataset.shape[0] // 3  # Determine the split size
                        splits = torch.split(train_dataset,
                                             [split_size, split_size, train_dataset.shape[0] - 2 * split_size])
                        decoded_parts = []
                        self.encoder_mlp.eval()
                        self.decoder_mlp.eval()
                        for part in splits:

                            encoded_part = (1.0 - self.a) * self.encoder_pod(
                                part.reshape(part.shape[0], -1)) + self.a * self.encoder_mlp(part)
                            decoded_part = (1.0 - self.b.view(1, 3, 1, 1, 1)) * self.decoder_pod(encoded_part).reshape(
                                part.shape[0],
                                *part.shape[
                                 1:]) + self.b.view(1, 3, 1, 1, 1) * self.decoder_mlp(
                                encoded_part)

                            decoded_parts.append(decoded_part)

                        decoded_train = torch.cat(decoded_parts, dim=0)
                        decoded_train_unnorm = decoded_train.detach().cpu().numpy() * std_value + mean_value
                        train_unnorm = train_dataset.detach().cpu().numpy() * std_value + mean_value

                        total_loss_train = self.loss_fn(decoded_train, train_dataset)
                        plt_loss.append(total_loss_train.detach().cpu().numpy())
                        plt_loss_test.append(total_loss_test)

                        np.save(
                            save_path + '/train_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            np.array(plt_loss))
                        np.save(save_path + '/test_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(plt_loss_test))
                        np.save(save_path + '/epochs_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(epoch_list))
                        np.save(save_path + '/a_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(a_list))
                        np.save(save_path + '/b_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(b_list))
                        np.save(
                            save_path + '/decoded_test_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            decoded_test_unnorm)
                        np.save(save_path + '/test_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                test_dataset_unnorm)
                        np.save(
                            save_path + '/decoded_train_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            decoded_train_unnorm)
                        np.save(
                            save_path + '/train_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            train_unnorm)
                        #torch.save({
                        #    'vhr_matrix': self.vhr,
                        #    'encoder_mlp_state_dict': self.encoder_mlp.state_dict(),
                        #    'decoder_mlp_state_dict': self.decoder_mlp.state_dict(),
                        #    'a_parameter': self.a,
                        #    'b_parameter': self.b
                        #}, os.path.join(save_path, f'{method}_parameters.pth'))
                        self.encoder_mlp.train()
                        self.decoder_mlp.train()
        if self.method == 'naive':
            self.optimizer = torch.optim.Adam([
                {'params': self.ParameterList, 'lr': lr},
            ])
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, min_lr=1e-5,
            #                                                        patience=100)
            for epoch in range(epochs):
                loss_epoch = 0.0
                for batch_data in dataloader:
                    batch_data = batch_data[0].to(self.device)
                    self.optimizer.zero_grad()
                    encoded_data = self.encoder_pod(batch_data.reshape(batch_data.shape[0], -1)) + self.encoder_mlp(
                        batch_data)
                    decoded_data = self.decoder_pod(encoded_data).reshape(encoded_data.shape[0],
                                                                          *batch_data.shape[1:]) + self.decoder_mlp(
                        encoded_data)
                    loss = self.loss_fn(decoded_data, batch_data)
                    loss.backward()
                    self.optimizer.step()
                    torch.nn.utils.clip_grad_norm_(self.ParameterList, max_norm=1.0)
                    loss_epoch += loss.detach().cpu().numpy()

                # scheduler.step(loss, epoch)
                loss_epoch = loss_epoch / len(dataloader)
                if (epoch) % 100 == 0:
                    with torch.no_grad():
                        total_loss, total_loss_test, decoded_test_unnorm, test_dataset_unnorm = self._print_stats(
                            train_dataset=None, test_dataset=torch.from_numpy(X_test).to(self.device), epoch=epoch,
                            loss=loss_epoch)
                        epoch_list.append(epoch)

                        train_dataset = torch.from_numpy(X_train).to(self.device)
                        split_size = train_dataset.shape[0] // 3  # Determine the split size
                        splits = torch.split(train_dataset,
                                             [split_size, split_size, train_dataset.shape[0] - 2 * split_size])
                        decoded_parts = []
                        self.encoder_mlp.eval()
                        self.decoder_mlp.eval()
                        for part in splits:
                            encoded_part = self.encoder_pod(
                                part.reshape(part.shape[0], -1)) + self.encoder_mlp(part)
                            decoded_part = self.decoder_pod(encoded_part).reshape(part.shape[0], *part.shape[
                                                                                                  1:]) + self.decoder_mlp(
                                encoded_part)
                            decoded_parts.append(decoded_part)

                        decoded_train = torch.cat(decoded_parts, dim=0)
                        decoded_train_unnorm = decoded_train.detach().cpu().numpy() * std_value + mean_value
                        train_unnorm = train_dataset.detach().cpu().numpy() * std_value + mean_value

                        total_loss_train = self.loss_fn(decoded_train, train_dataset)
                        plt_loss.append(total_loss_train.detach().cpu().numpy())
                        plt_loss_test.append(total_loss_test)

                        np.save(
                            save_path + '/train_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            np.array(plt_loss))
                        np.save(save_path + '/test_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(plt_loss_test))
                        np.save(save_path + '/epochs_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(epoch_list))
                        np.save(save_path + '/a_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(a_list))
                        np.save(save_path + '/b_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                np.array(b_list))
                        np.save(
                            save_path + '/decoded_test_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            decoded_test_unnorm)
                        np.save(save_path + '/test_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                                test_dataset_unnorm)
                        np.save(
                            save_path + '/decoded_train_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            decoded_train_unnorm)
                        np.save(
                            save_path + '/train_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
                            train_unnorm)
                        #torch.save({
                        #    'vhr_matrix': self.vhr,
                        #    'encoder_mlp_state_dict': self.encoder_mlp.state_dict(),
                        #    'decoder_mlp_state_dict': self.decoder_mlp.state_dict()
                        #}, os.path.join(save_path, f'{method}_parameters.pth'))
                        self.encoder_mlp.train()
                        self.decoder_mlp.train()

            a_list.append(-99)
            b_list.append(-99)

        return plt_loss, plt_loss_test, epoch_list, decoded_test_unnorm, test_dataset_unnorm, a_list, b_list

    def _visualize_reconstruction(self, decoded_test, test_dataset, epoch):

        decoded_test_unnorm = decoded_test.detach().cpu().numpy() * std_value + mean_value
        test_dataset_unnorm = test_dataset.detach().cpu().numpy() * std_value + mean_value
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Calculate the min and max values across both slices
        vmin = min(decoded_test_unnorm[0, 0, :, :, 0].min(), test_dataset_unnorm[0, 0, :, :, 0].min())
        vmax = max(decoded_test_unnorm[0, 0, :, :, 0].max(), test_dataset_unnorm[0, 0, :, :, 0].max())

        # Plot the first slice
        im1 = axes[0].imshow(decoded_test_unnorm[0, 0, :, :, 0], cmap='seismic', vmin=vmin, vmax=vmax)
        axes[0].set_title('Decoded')
        fig.colorbar(im1, ax=axes[0], orientation='vertical')

        # Plot the second slice
        im2 = axes[1].imshow(test_dataset_unnorm[0, 0, :, :, 0], cmap='seismic', vmin=vmin, vmax=vmax)
        axes[1].set_title('Ground Truth')
        fig.colorbar(im2, ax=axes[1], orientation='vertical')

        # Display the plot
        plt.savefig(save_path + '/figures/' + str(epoch) + '.png')
        plt.close()
        return decoded_test_unnorm, test_dataset_unnorm

    def _print_stats(self, train_dataset, test_dataset, epoch, loss):

        if self.method == 'pod-ae-tunable':
            # encoded_train = (1.0 - self.a) * self.encoder_pod(
            #    train_dataset.reshape(train_dataset.shape[0], -1)) + self.a * self.encoder_mlp(train_dataset)
            # decoded_train = (1.0 - self.b) * self.decoder_pod(encoded_train).reshape(
            #    encoded_train.shape[0], *train_dataset.shape[1:]) + self.b * self.decoder_mlp(encoded_train)
            total_loss = loss  # self.loss_fn(train_dataset, decoded_train)

            split_size = test_dataset.shape[0] // 3  # Determine the split size
            splits = torch.split(test_dataset, [split_size, split_size, test_dataset.shape[0] - 2 * split_size])
            decoded_parts = []
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            for part in splits:
                encoded_part = (1.0 - self.a) * self.encoder_pod(
                    part.reshape(part.shape[0], -1)) + self.a * self.encoder_mlp(part)
                decoded_part = (1.0 - self.b.view(1, 3, 1, 1, 1)) * self.decoder_pod(encoded_part).reshape(
                    part.shape[0], *part.shape[
                                    1:]) + self.b.view(1, 3, 1, 1, 1) * self.decoder_mlp(
                    encoded_part)
                decoded_parts.append(decoded_part)
            
            decoded_test = torch.cat(decoded_parts, dim=0)
            total_loss_test = self.loss_fn(test_dataset, decoded_test)

            print(f'Epoch: {epoch}, Total train Loss: {total_loss:.5e}')
            print(f'Epoch: {epoch}, Total test Loss: {total_loss_test:.5e}')
            print(f'Epoch: {epoch}, a: {torch.norm(self.a).detach().cpu().numpy():.5e}')
            print(f'Epoch: {epoch}, b: {torch.norm(self.b).detach().cpu().numpy():.5e}')
            self.encoder_mlp.train()
            self.decoder_mlp.train()
        if self.method == 'naive':
            # encoded_train = self.encoder_pod(
            #    train_dataset.reshape(train_dataset.shape[0], -1)) + self.encoder_mlp(train_dataset)
            # decoded_train = self.decoder_pod(encoded_train).reshape(
            #    encoded_train.shape[0], *train_dataset.shape[1:]) + self.decoder_mlp(encoded_train)
            total_loss = loss  # self.loss_fn(train_dataset, decoded_train)

            split_size = test_dataset.shape[0] // 3  # Determine the split size
            splits = torch.split(test_dataset, [split_size, split_size, test_dataset.shape[0] - 2 * split_size])
            decoded_parts = []
            self.encoder_mlp.eval()
            self.decoder_mlp.eval()
            for part in splits:
                encoded_part = self.encoder_pod(part.reshape(part.shape[0], -1)) + self.encoder_mlp(part)
                decoded_part = self.decoder_pod(encoded_part).reshape(part.shape[0],
                                                                      *part.shape[1:]) + self.decoder_mlp(encoded_part)
                decoded_parts.append(decoded_part)

            decoded_test = torch.cat(decoded_parts, dim=0)
            total_loss_test = self.loss_fn(test_dataset, decoded_test)

            print(f'Epoch: {epoch}, Total train Loss: {total_loss:.5e}')
            print(f'Epoch: {epoch}, Total test Loss: {total_loss_test:.5e}')
            self.encoder_mlp.train()
            self.decoder_mlp.train()

        decoded_test_unnorm, test_dataset_unnorm = self._visualize_reconstruction(decoded_test, test_dataset, epoch)

        return total_loss, total_loss_test.item(), decoded_test_unnorm, test_dataset_unnorm


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

data_path1 = f'/isotropic{grid}coarse_1.h5'
data_path2 = f'/isotropic{grid}coarse_2.h5'
hdf5file1 = h5py.File(data_path1,'r')
hdf5file2 = h5py.File(data_path2,'r')
velocity_data = np.zeros((hdf5file1.keys().__len__()-3 + hdf5file2.keys().__len__()-3, 3, grid, grid, grid))
tstep = 0
for key in hdf5file1.keys():
    if key not in ['xcoor', 'ycoor', 'zcoor']:
        data = np.array(hdf5file1[key])
        velocity_data[tstep, 0] = data[:,:,:,0]
        velocity_data[tstep, 1] = data[:, :, :, 1]
        velocity_data[tstep, 2] = data[:, :, :, 2]
        tstep +=1

for key in hdf5file2.keys():
    if key not in ['xcoor', 'ycoor', 'zcoor']:
        data = np.array(hdf5file2[key])
        velocity_data[tstep, 0] = data[:,:,:,0]
        velocity_data[tstep, 1] = data[:, :, :, 1]
        velocity_data[tstep, 2] = data[:, :, :, 2]
        tstep +=1

velocity_data = np.float32(velocity_data)
num_samples = velocity_data.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Define split size
split_index = int(0.7 * num_samples)

X_train = velocity_data[::2]
X_test = velocity_data[1::2]

mean_value = np.mean(X_train, axis=0)
std_value = np.std(X_train, axis=0)


X_train_normalized = (X_train - mean_value) / (std_value)
X_test_normalized = (X_test - mean_value) / (std_value)


batch_size = 20
normalize = False
#pochs = 4000
learning_rate = 1e-4  # 0.001

ae = AutoEncoder(rank, method, normalize)
ae._init_encoder_decoder(X_train_normalized)
plt_loss, plt_loss_test, epochs, decoded_test_unnorm, test_dataset_unnorm, a_list, b_list = ae._fit(X_train_normalized,
                                                                                                    X_test_normalized,
                                                                                                    batch_size, epochs,
                                                                                                    learning_rate)
np.save(save_path + '/train_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method, np.array(plt_loss))
np.save(save_path + '/test_loss_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method,
        np.array(plt_loss_test))
np.save(save_path + '/epochs_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method, np.array(epochs))
np.save(save_path + '/a_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method, np.array(a_list))
np.save(save_path + '/b_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method, np.array(b_list))
np.save(save_path + '/decoded_test_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method, decoded_test_unnorm)
np.save(save_path + '/test_data_cnn_' + str(grid) + '_' + str(encoded_space_dim) + '_' + method, test_dataset_unnorm)

