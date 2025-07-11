import numpy as np
import matplotlib.pyplot as plt
from koopman import koopman, convolutional_mse_wave
import torch
np.random.seed(1990)
torch.random.manual_seed(1990)
# Define time and spatial range
t = np.linspace(0, 100000, 100000)  # 1000 time steps
u = np.linspace(1, 256, 256)  # Spatial index from 1 to 256


U, T = np.meshgrid(u, t, indexing='ij')
u_mean_t = (np.sin(0.01 * t) + 1) * 100 + 28  # Mean location of the wave over time
sigmaS = 10  # Controls the width of the traveling wave
traj = np.exp(-((U - u_mean_t)**2) / (2 * sigmaS)).T
freqs = 1
u, s, vh = np.linalg.svd(traj[:50000], full_matrices=False)
vhr = vh[:2*freqs, :]
method='ae'
traj = np.expand_dims(traj, axis=1)
k = koopman(convolutional_mse_wave(num_freqs=freqs, n=512, vhr=vhr, method=method, final_channel=32), device='cuda', batch_size=1280)
k.fit(traj[:50000], iterations = 500, interval = 5, verbose=True)

xhat_koopman = k.predict(100000)

print('Training MSE :', np.mean((traj[:50000] - xhat_koopman[:50000])**2))
print('Testing MSE :', np.mean((traj[50000:] - xhat_koopman[50000:])**2))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(traj[-1000:,0].T, aspect='auto', cmap='viridis')
axes[0].set_title("Original Data (x)")

# Plot xhat_koopman (Koopman approximation)
axes[1].imshow(xhat_koopman[-1000:,0].T, aspect='auto', cmap='viridis')
axes[1].set_title("Koopman Approximation (x̂_koopman)")

# Display plots
plt.show()

method='naive'
np.save(f'./GT_{method}.npy', traj)
np.save(f'./pred_{method}.npy', xhat_koopman)
print('here')

