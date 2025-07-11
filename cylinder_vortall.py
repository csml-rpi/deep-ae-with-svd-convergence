import matplotlib.pyplot as plt
from fourier_koopman import koopman, convolutional_mse, convolutional_mse_cylinder
import numpy as np
import torch
import h5py
import scipy
from soap import SOAP
import argparse

np.random.seed(123)
torch.manual_seed(123)

# Initialize argument parser
parser = argparse.ArgumentParser(description='Autoencoder Arguments')

# Add arguments

parser.add_argument('--method', type=str, default='naive', choices=['ae', 'pod', 'pod-ae', 'tunable', 'naive'],
                    help='Method for autoencoder')
parser.add_argument('--final', type=int, default=64,
                    help='final channel size')


# Parse arguments
args = parser.parse_args()

# Access arguments

method = args.method
final_channel = args.final



traj = np.load('./vort_all_data.npy')
mean = np.mean(traj[:100], axis=(0,1,2,3), keepdims=True)
std = np.std(traj[:100], axis=(0,1,2,3), keepdims=True)
traj = (traj-mean)/std
freqs = 2
u, s, vh = np.linalg.svd(traj[:100].reshape(traj[:100].shape[0], -1), full_matrices=False)
vhr = vh[:2*freqs, :]

k = koopman(convolutional_mse_cylinder(num_freqs=freqs, n=512, vhr=vhr, method=method, final_channel=final_channel), device='cuda', batch_size=16)
k.fit(traj[:100], iterations = 1000, interval = 5, verbose=True)

xhat_koopman = k.predict(151)
np.save(f'./vort_koopman_{method}_{final_channel}.npy', xhat_koopman)
print('Training MSE :', np.mean((traj[:100] - xhat_koopman[:100])**2))
print('Testing MSE :', np.mean((traj[100:] - xhat_koopman[100:])**2))
np.savez(f'./loss_vals_{method}_{final_channel}', train_loss = np.mean((traj[:100] - xhat_koopman[:100])**2),
         test_loss = np.mean((traj[100:] - xhat_koopman[100:])**2))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow((traj  *std + mean)[-1, 0, :, :], aspect='auto', cmap='coolwarm')
axes[0].set_title("Original Data (x)")

# Plot xhat_koopman (Koopman approximation)
axes[1].imshow((xhat_koopman * std + mean)[-1, 0, :, :], aspect='auto', cmap='coolwarm')
axes[1].set_title("Koopman Approximation (x̂_koopman)")

# Display plots

plt.savefig(f'./vort_koopman_{method}_{final_channel}.png')

# print('here')