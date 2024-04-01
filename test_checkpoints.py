from matplotlib import pyplot as plt
import numpy as np
from schema import schema
from kalman_filter import *
from models import *
from plotting import *
from trajectories import *
from quinine import QuinineArgumentParser
from train import *

rng = np.random.default_rng(seed=0)

def main():
    matrix_dim = 2
    matrix_type = None
    optim='adamw'
    num_tests = 101
    traj_len = 256
    lr = 0.01
    family='gpt'

    save_path = './results/.'
    A = np.loadtxt("./matrices/" + str(matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(matrix_type) if matrix_type is not None else "") + "/A.out", delimiter=',')
    B = np.expand_dims(np.loadtxt("./matrices/" + str(matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(matrix_type) if matrix_type is not None else "") + "/B.out", delimiter=','), axis=1) # Not used, should be zero
    C = np.expand_dims(np.loadtxt("./matrices/" + str(matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(matrix_type) if matrix_type is not None else "") + "/C.out", delimiter=','), axis=0)
    state_dim, obs_dim = A.shape[0], C.shape[0]
    
    noise_block = np.loadtxt("./matrices/" + str(matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(matrix_type) if matrix_type is not None else "") + "/noise_block.out", delimiter=',')
    Q, R, S = noise_block[0:state_dim, 0:state_dim], noise_block[state_dim:state_dim+obs_dim, state_dim:state_dim+obs_dim], noise_block[0:state_dim, state_dim:state_dim+obs_dim] # S should be zero
    x0 = rng.normal(loc=0.0, scale=1.0, size=state_dim)      

        # plt.figure()
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.xlabel('Median Error')
        # plt.ylabel('CCDF')
        # plt.title("Complementary CDF of Median Errors for Checkpoints")

    start_states = np.zeros(shape=(num_tests, state_dim))
    for i in range(num_tests): start_states[i] = x0

    x, y = generate_traj(num_tests, traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)

    baseline = KfModel(A=A, C=C, Q=Q, R=R, start_states=start_states)
    errors_baseline = evaluate_nextstate_model(baseline, y)

    # scale = np.array(range(args.testing.num_tests))/args.testing.num_tests
    
    # lr_sweep = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    size_sweep = ['tiny', 'small', 'medium', 'large']

    plt.figure()
    plt.title("Median gaps vs Training Iteration Checkpoint")
    
    for size in size_sweep:
        median_gaps = []
        checkpoint_nums = range(20, 620, 20)

        if size == 'tiny':
            n_positions, n_embd, n_layer, n_head = 1024, 32, 2, 2
        elif size == 'small':
            n_positions, n_embd, n_layer, n_head = 1024, 64, 3, 2
        elif size == 'medium':
            n_positions, n_embd, n_layer, n_head = 1024, 128, 6, 4
        elif size == 'large':
            n_positions, n_embd, n_layer, n_head = 1024, 256, 12, 8
            checkpoint_nums = range(20, 300, 20)
        else:
            print("Invalid Size")

        for ch_num in checkpoint_nums:
            state_path = os.path.join(save_path, '{0}_{1}_order{2}_lr{3}chpt{4}.pt'.format(family, size, matrix_dim, lr, ch_num))
            print("state_path", state_path)

            
            model = GPTModel(n_dims_obs=obs_dim, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)
            optimizer =  {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}.get(optim)(model.parameters())
            load_model_and_optimizer(state_path, model, optimizer)

            model.eval()

            errors = evaluate_nextstate_model(model, y)
            gap_values = [errors[i] - errors_baseline[i] for i in range(num_tests)]

            median_gaps.append(np.median(gap_values))

            freq = np.sort(gap_values)
            # plt.plot(freq, scale, label='cdf {0}'.format(ch_num))
            # plt.plot(freq, np.ones_like(scale) - scale, label='Iteration {0}'.format(ch_num))

        # plt.legend()
        # plt.show()

        print("median gaps, size {0}".format(size), median_gaps)
        plt.plot(checkpoint_nums, median_gaps, label="size {0}".format(size))

    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
