import numpy as np
import random
from network import TsodyksHopfieldNetwork, create_pattern, hamming_distance
import matplotlib.pyplot as plt
from scipy.spatial import distance


def test_Tsodyks():
    num_neurons = 80
    num_morphs = 30
    num_iterations = 10 
    eta = 0.1

    x0, x1, x_morph = create_pattern(num_neurons, num_morphs)
    w_morph = np.zeros((num_morphs, ))
    x_init = np.array([x0, x1])
    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    correlations_gradual = np.zeros((num_iterations, num_morphs))
    correlations_random = np.zeros((num_iterations, num_morphs))
    
    ########################## gradual morphing ##########################
    network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
    network.train_weights(x=x_init, w=w_init) # pre-training of two contexts

    for i in range(num_iterations):
        for j in range(num_morphs):
            x_one_step = network.one_step_dynamics(x_morph[j])
            hamming_distance_temp = hamming_distance(x_one_step, x_morph[j])/hamming_distance_init
            w_morph[j] += eta * hamming_distance_temp
            network.train_weights(x_morph[j][None], w=w_morph[[j]])
            
        for j in range(num_morphs):
            converged_state = network.converge(x_morph[j])
            correlations_gradual[i, j] = np.sum(converged_state[0] * x_init[0]) / num_neurons

    ########################### random morphing ##########################
    network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
    network.train_weights(x=x_init, w=w_init)     

    for i in range(num_iterations):
        random_perms = np.random.permutation(np.arange(num_morphs))

        for j in range(num_morphs):
            x_one_step = network.one_step_dynamics(x_morph[random_perms[j]])
            hamming_distance_temp = hamming_distance(x_one_step, x_morph[random_perms[j]]) / hamming_distance_init
            w_morph[random_perms[j]] += eta * hamming_distance_temp
            network.train_weights(x_morph[random_perms[j]][None], w=w_morph[[random_perms[j]]])
        
        for j in range(num_morphs):
            converged_state = network.converge(x_morph[j])
            correlations_random[i, j] = np.sum(converged_state[0] * x_init[0]) / num_neurons 
    
    return correlations_gradual, correlations_random


if __name__=="__main__":
    # network = test_classical_HN()
    for _ in range (10):
        correlations_gradual, correlations_random = test_Tsodyks()
        fig, ax = plt.subplots(2,5, figsize=(15,7))
        for i in range (10):
            ax[i//5, i%5].plot(correlations_gradual[i], '--', label='Gradual')
            ax[i//5, i%5].plot(correlations_random[i], label='Random')
            ax[i//5, i%5].set_title('Session_{}'.format(i))
        fig.legend()
        plt.show()