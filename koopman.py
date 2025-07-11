#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Henning Lange (helange@uw.edu)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from scipy.ndimage import gaussian_filter


class koopman(nn.Module):
    
    r'''
    
    model_obj: dobject that specifies the function f and how to optimize
               it. The object needs to implement numerous function. See
               below for some examples.
               
    sample_num: number of samples from temporally local loss used to 
                reconstruct the global error surface.
                
    batch_size: Number of temporal snapshots processed by SGD at a time
                default = 32
                type: int
        
    parallel_batch_size: Number of temporaly local losses sampled in parallel. 
                         This number should be as high as possible but low enough
                         to not cause memory issues.
                         default = 1000
                         type: int
                
    device: The device on which the computations are carried out.
            Example: cpu, cuda:0, or list of GPUs for multi-GPU usage, i.e. ['cuda:0', 'cuda:1']
            default = 'cpu'
            
        
    '''
    
    
    def __init__(self, model_obj, sample_num = 12, **kwargs):
        
        
        super(koopman, self).__init__()
        self.num_freq = model_obj.num_freq
    
    
        if 'device' in kwargs:
            self.device = kwargs['device']
            if type(kwargs['device']) == list:
                self.device = kwargs['device'][0]
                multi_gpu = True
            else:
                multi_gpu = False
        else:
            self.device = 'cpu'
            multi_gpu = False
            
        #Inital guesses for frequencies
        if self.num_freq == 1:
            self.omegas = torch.tensor([0.2], device = self.device)
        else:
            self.omegas = torch.linspace(0.01,0.5,self.num_freq, device = self.device)
            
        self.multi_gpu = multi_gpu
            
            
        self.parallel_batch_size = kwargs['parallel_batch_size'] if 'parallel_batch_size' in kwargs else 1000
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
            
        model_obj = model_obj.to(self.device)
        self.model_obj = nn.DataParallel(model_obj, device_ids= kwargs['device']) if multi_gpu else model_obj
            
        self.sample_num = sample_num

        
        
        
    def sample_error(self, xt, which):
        '''
        
        sample_error computes all temporally local losses within the first
        period, i.e. between [0,2pi/t]

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        i : TYPE int
            Index of the entry of omega

        Returns
        -------
        TYPE numpy.array
            Matrix that contains temporally local losses between [0,2pi/t]
            dimensions: [T, sample_num]

        '''
        
        num_samples = self.sample_num
        omega = self.omegas
        
        if type(xt) == np.ndarray:
            xt = torch.tensor(xt, device = self.device)
            
        t = torch.arange(xt.shape[0], device=self.device)+1
        
        errors = []
        batch = self.parallel_batch_size
        pi_block = torch.zeros((num_samples, len(omega)), device=self.device)
        pi_block[:, which] = torch.arange(0,num_samples)*np.pi*2/num_samples
        
        for i in range(int(np.ceil(xt.shape[0]/batch))):
            t_batch = t[i*batch:(i+1)*batch][:,None]
            wt = t_batch*omega[None]
            wt[:, which] = 0
            wt = wt[:,None] + pi_block[None]
            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)
            loss = self.model_obj(k, xt[i*batch:(i+1)*batch, None]).cpu().detach().numpy()
            errors.append(loss)
            
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        return np.concatenate(errors, axis=0)
    
    
    def reconstruct(self, errors, use_heuristic = True):
        
        e_fft = np.fft.fft(errors)
        E_ft = np.zeros(errors.shape[0]*self.sample_num, dtype=np.complex64)
        
        for t in range(1,e_fft.shape[0]+1):
            E_ft[np.arange(self.sample_num//2)*t] += e_fft[t-1,:self.sample_num//2]
            
        E_ft = np.concatenate([E_ft, np.conj(np.flip(E_ft, -1))])[:-1]
        E = np.real(np.fft.ifft(E_ft))
        
        if use_heuristic:
            E = -np.abs(E-np.median(E))
            #E = gaussian_filter(E, 5)
            
        return E, E_ft
    
    
    def fft(self, xt, i, verbose=False):
        '''
        
        fft first samples all temporaly local losses within the first period
        and then reconstructs the global error surface w.r.t. omega_i
        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        i : TYPE int
            Index of the entry of omega
        verbose : TYPE boolean, optional
            DESCRIPTION. The default is False.
        Returns
        -------
        E : TYPE numpy.array
            Global loss surface in time domain.
        E_ft : TYPE
            Global loss surface in frequency domain.
        '''
        
        E, E_ft = self.reconstruct(self.sample_error(xt, i))
        omegas = np.linspace(0,1,len(E))
        
        idxs = np.argsort(E[:len(E_ft)//2])
        
        omegas_actual = self.omegas.cpu().detach().numpy()
        omegas_actual[i] = -1
        found = False
        
        j=0
        while not found:
            # The if statement avoids non-unique entries in omega and that the
            # frequencies are 0 (should be handle by bias term)
            if idxs[j]>1 and np.all(np.abs(2*np.pi/omegas_actual - 1/omegas[idxs[j]])>1):
                found = True
                # if verbose:
                #     print('Setting ',i,'to',1/omegas[idxs[j]])
                self.omegas[i] = torch.from_numpy(np.array([omegas[idxs[j]]]))
                self.omegas[i] *= 2*np.pi
            
            j+=1
            
        return E, E_ft
    
    
    
    
    def sgd(self, xt, verbose=False):
        '''
        
        sgd performs a single epoch of stochastic gradient descent on parameters
        of f (Theta) and frequencies omega

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        verbose : TYPE boolean, optional
            The default is False.

        Returns
        -------
        TYPE float
            Loss.

        '''
        
        batch_size = self.batch_size
        
        T = xt.shape[0]
        
        omega = nn.Parameter(self.omegas)

        opt = optim.Adam(self.model_obj.parameters(), lr=3e-4)
        opt_omega = optim.Adam([omega], lr=1e-5)
        
        
        T = xt.shape[0]
        t = torch.arange(T, device=self.device)
        
        losses = []
        
        for i in range(len(t)//batch_size):
            
            ts = t[i*batch_size:(i+1)*batch_size]
            o = torch.unsqueeze(omega, 0)
            ts_ = torch.unsqueeze(ts,-1).type(torch.get_default_dtype()) + 1
            
            xt_t = torch.tensor(xt[ts.cpu().numpy(),:], device=self.device)
            
            wt = ts_*o
            
            k = torch.cat([torch.cos(wt), torch.sin(wt)], -1)            
            loss = torch.mean(self.model_obj(k, xt_t))
            
            opt.zero_grad()
            opt_omega.zero_grad()
            
            loss.backward()
            
            opt.step()
            opt_omega.step()
            
            losses.append(loss.cpu().detach().numpy())
            
        # if verbose:
        #     print('Setting to', 2*np.pi/omega)
            
        self.omegas = omega.data
                

        return np.mean(losses)
    
    
    
    def fit(self, xt, iterations = 10, interval = 5, cutoff = np.inf, verbose=False):
        '''
        Given a dataset, this function alternatingly optimizes omega and 
        parameters of f. Specifically, the algorithm performs interval many
        epochs, then updates all entries in omega. This process is repeated
        until iterations-many epochs have been performed

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data whose first dimension is time.
        iterations : TYPE int, optional
            Total number of SGD epochs. The default is 10.
        interval : TYPE, optional
            The interval at which omegas are updated, i.e. if 
            interval is 5, then omegas are updated every 5 epochs. The default is 5.
        verbose : TYPE boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
    
        assert(len(xt.shape) > 1), 'Input data needs to be at least 2D'
        losses = []
        for i in range(iterations):
            
            if i%interval == 0 and i < cutoff: 
                for k in range(self.num_freq):
                    self.fft(xt, k, verbose=verbose)
            
            if verbose:
                print('Iteration ',i)
                # print(2*np.pi/self.omegas)
            
            l = self.sgd(xt, verbose=verbose)
            if verbose:
                print('Loss: ',l)
                print('b : ', self.model_obj.b.item())
            losses.append(l)
            
        return losses
            
            
            
    def predict(self, T):
        '''
        Predicts the data from 1 to T.

        Parameters
        ----------
        T : TYPE int
            Prediction horizon

        Returns
        -------
        TYPE numpy.array
            xhat from 0 to T.

        '''
        
        t = torch.arange(T, device=self.device)+1
        ts_ = torch.unsqueeze(t,-1).type(torch.get_default_dtype())

        o = torch.unsqueeze(self.omegas, 0)
        k = torch.cat([torch.cos(ts_*o), torch.sin(ts_*o)], -1)
        
        if self.multi_gpu:
            mu = self.model_obj.module.decode_predict(k)
        else:
            mu = self.model_obj.decode_predict(k)
        

        return mu.cpu().detach().numpy()





class model_object(nn.Module):
    
    def __init__(self, num_freq):
        super(model_object, self).__init__()
        self.num_freq = num_freq
        
    
    
    def forward(self, y, x):
        '''
        Forward computes the error.
        
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
                
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        '''
        
        
        raise NotImplementedError()
    
    def decode(self, y):
        '''
        Evaluates f at temporal snapshots y
        
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
                
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        '''
        raise NotImplementedError()




class convolutional_mse(model_object):
    
    
    def __init__(self, num_freqs, n, vhr, method):
        super(convolutional_mse, self).__init__(num_freqs)

        self.fc_layers = nn.Sequential(
            nn.Linear(2*num_freqs, 256),  # First hidden layer
            nn.GELU(),
            nn.Linear(256, 512),  # Second hidden layer
            nn.GELU(),
            nn.Linear(512, 512 * 8 * 8)  # Latent dimension
        )


        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(128, 2, kernel_size=5, stride=1, padding=2)
        self.b = nn.Parameter(torch.zeros(1))
        self.vhr = torch.tensor(vhr, dtype=torch.float32)
        self.method = method

    def decode(self, x):

        x = self.fc_layers(x).view(-1, 512, 8, 8)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = nn.GELU()(self.deconv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = nn.GELU()(self.deconv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)
        
        return x

    def decode_predict(self,x):

        if self.method=='ae':
            xhat = self.decode(x)
        elif self.method=='tunable':
            xhat = self.b * self.decode(x) + ((1-self.b) * x @ self.vhr.to(x.device)).reshape(x.shape[0], 2, 64, 64)
        return xhat
        
    def forward(self, y, x):
        y_dim = y.dim()
        if y_dim==3:
            y_dim1 = y.shape[0]
            y_dim2 = y.shape[1]
            y_dim3 = y.shape[2]
            y = y.reshape(y_dim1 * y_dim2, y_dim3)
        if self.method=='ae':
            xhat = self.decode(y)
        elif self.method=='tunable':
            xhat = self.b * self.decode(y) + ((1-self.b) * y @ self.vhr.to(y.device)).reshape(y.shape[0], 2, 64, 64)

        if y_dim == 3:
            xhat = xhat.reshape(y_dim1, y_dim2, 2, 64, 64)
        return torch.mean((xhat-x)**2, dim=(-1, -2, -3))


class convolutional_mse_wave(model_object):

    def __init__(self, num_freqs, n, vhr, method, final_channel):
        super(convolutional_mse_wave, self).__init__(num_freqs)

        self.final_channel = final_channel
        self.fc = nn.Sequential(
            
            nn.Linear(2*num_freqs, final_channel * (256 // 16))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(final_channel, final_channel//2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),  # F.silu equivalent
            nn.ConvTranspose1d(final_channel//2, final_channel//4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(final_channel//4, final_channel//8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(final_channel//8, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        self.b = nn.Parameter(torch.zeros(1))
        self.vhr = torch.tensor(vhr, dtype=torch.float32)
        self.method = method

    def decode(self, x):

        x = self.fc(x).view(-1, self.final_channel, 256//16)
        x = self.decoder(x)

        return x

    def decode_predict(self, x):

        if self.method == 'ae':
            xhat = self.decode(x)
        elif self.method == 'tunable':
            xhat = self.b * self.decode(x) + ((1 - self.b) * x @ self.vhr.to(x.device)).reshape(x.shape[0], 1, 256)
        elif self.method=='pod':
            xhat = (x @ self.vhr.to(x.device)).reshape(-1, 1, 256)
        elif self.method == 'naive':
            xhat = self.decode(x) + (x @ self.vhr.to(x.device)).reshape(-1, 1, 256)

        return xhat

    def forward(self, y, x):
        y_dim = y.dim()
        if y_dim == 3:
            y_dim1 = y.shape[0]
            y_dim2 = y.shape[1]
            y_dim3 = y.shape[2]

            y = y.reshape(y_dim1 * y_dim2, y_dim3)
        if self.method == 'ae':
            xhat = self.decode(y)
        elif self.method == 'tunable':
            xhat = self.b * self.decode(y) + ((1 - self.b) * y @ self.vhr.to(y.device)).reshape(-1, 1, 256)
        elif self.method=='pod':
            xhat = (y @ self.vhr.to(y.device)).reshape(-1, 1, 256)
        elif self.method == 'naive':
            xhat = self.decode(y) + (y @ self.vhr.to(y.device)).reshape(-1, 1, 256)

        if y_dim == 3:
            xhat = xhat.reshape(y_dim1, y_dim2, 1, 256)
        return torch.mean((xhat - x) ** 2, dim=(-1, -2))

class convolutional_mse_cylinder(model_object):

    def __init__(self, num_freqs, n, vhr, method, final_channel):
        super(convolutional_mse_cylinder, self).__init__(num_freqs)

        self.final_channel = final_channel
        self.fc = nn.Sequential(
            
            nn.Linear(2*num_freqs, final_channel * (256 // 16) * (512//16))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(final_channel, final_channel//2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),  # F.silu equivalent
            nn.ConvTranspose2d(final_channel//2, final_channel//4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(final_channel//4, final_channel//8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(final_channel//8, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        self.b = nn.Parameter(torch.zeros(1))
        self.vhr = torch.tensor(vhr, dtype=torch.float32)
        self.method = method

    def decode(self, x):

        x = self.fc(x).view(-1, self.final_channel, 256//16, 512//16)
        x = self.decoder(x)

        return x

    def decode_predict(self, x):

        if self.method == 'ae':
            xhat = self.decode(x)
        elif self.method == 'tunable':
            xhat = self.b * self.decode(x) + ((1 - self.b) * x @ self.vhr.to(x.device)).reshape(x.shape[0], 1, 256, 512)
        elif self.method=='pod':
            xhat = (x @ self.vhr.to(x.device)).reshape(-1, 1, 256, 512)
        elif self.method == 'naive':
            xhat = self.decode(x) + (x @ self.vhr.to(x.device)).reshape(-1, 1, 256, 512)

        return xhat

    def forward(self, y, x):
        y_dim = y.dim()
        if y_dim == 3:
            y_dim1 = y.shape[0]
            y_dim2 = y.shape[1]
            y_dim3 = y.shape[2]

            y = y.reshape(y_dim1 * y_dim2, y_dim3)
        if self.method == 'ae':
            xhat = self.decode(y)
        elif self.method == 'tunable':
            xhat = self.b * self.decode(y) + ((1 - self.b) * y @ self.vhr.to(y.device)).reshape(-1, 1, 256, 512)
        elif self.method=='pod':
            xhat = (y @ self.vhr.to(y.device)).reshape(-1, 1, 256, 512)
        elif self.method == 'naive':
            xhat = self.decode(y) + (y @ self.vhr.to(y.device)).reshape(-1, 1, 256, 512)

        if y_dim == 3:
            xhat = xhat.reshape(y_dim1, y_dim2, 1, 256, 512)
        return torch.mean((xhat - x) ** 2, dim=(-1, -2, -3))
