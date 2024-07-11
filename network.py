from typing import Optional, Union
import numpy as np
import torch.nn as nn
import random

class TsodyksHopfieldNetwork(nn.Module):
    def __init__(self, N_neurons, num_iter, eta, threshold):
        super(TsodyksHopfieldNetwork, self).__init__()
        self.num_neurons = N_neurons
        self.W = np.zeros((N_neurons, N_neurons))
        self.scaled_W = np.zeros((N_neurons, N_neurons))
        self.num_iter = num_iter
        self.threshold = threshold
        self.N = 0
        self.running_rho = 0.0
        self.eta = eta
        
    def train_weights(self, x: np.ndarray, w: Optional[np.ndarray] = None):
        N, D = x.shape
        assert self.num_neurons == D
        
        if w is None:
            w = np.ones((N, ))
        rho = np.mean(x)
        
        for i in range(N): #N_neurons
            t = x[i] - rho
            self.W += w[i] * np.outer(t, t)

        np.fill_diagonal(self.W,0)
        self.scaled_W = self.W / N
        
    def one_step_dynamics(self, x: np.ndarray):
        return np.sign(self.scaled_W.dot(x) - self.threshold)
        
    def converge(self, x: np.ndarray):
        x_old = x # input pattern
        for i in range(self.num_iter):
            x = np.sign(self.scaled_W.dot(x) - self.threshold)
            if np.all(x == x_old):
                break
            x_old = x
        return x, i
    

def hamming_distance(x1: np.ndarray, x2: np.ndarray):
    assert len(x1) == len(x2)
    return (len(x1) - np.sum(np.equal(x1, x2)))/len(x1)
        
def create_pattern(num_neurons, num_morphs):
    x0 = (np.random.binomial(1, 0.5, num_neurons) - 0.5) * 2
    x1 = (np.random.binomial(1, 0.5, num_neurons) - 0.5) * 2

    different_inds = np.where(x0 != x1)[0]

    random.shuffle(different_inds)
    morph_inds = np.split(different_inds, np.sort(np.random.choice(len(different_inds), num_morphs, replace=False)))[1:]

    x_morph = [x0]
    for i in range(num_morphs):
        x_morph_temp = x_morph[-1].copy()
        x_morph_temp[morph_inds[i]] = -x_morph_temp[morph_inds[i]]
        x_morph.append(x_morph_temp)

    x_morph = np.array(x_morph[1:])
    return x0, x1, x_morph


############################## Ignore: extra functions ##############################
def random_idx (steps, ratio):
    random_perms_temp = np.random.permutation(np.arange(steps))
    random_perms = []
    idx = 5

    random_perms.append(0)
    for i in range (1,len(random_perms_temp)-1):
        if i // ratio == 0:
            random_perms.append(np.arange(steps)[idx:idx+5])
            if idx <= 24: 
                idx+= 3
        else: 
            random_perms.append(random_perms_temp[i])
    random_perms.append(len(random_perms_temp))

    random_perms = np.hstack(random_perms)
    return random_perms

def create_pattern2(pattern_length, N_neuron):

    # Define the initial and final patterns
    pattern_a = np.array([1] * N_neuron)
    pattern_b = np.array([-1] * N_neuron)

    # Number of steps (rows in the matrix)
    steps = pattern_length

    # Create the matrix
    matrix = np.zeros((steps, len(pattern_a)))

    # Fill the matrix with the transition patterns
    for i in range(steps):
        transition_pattern = pattern_a.copy()
        num_changes = int((i + 1) / steps * len(pattern_a))
        for j in range(num_changes):
            transition_pattern[j] = pattern_b[j]
        matrix[i] = transition_pattern
    
    return pattern_a, pattern_b, matrix