import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from quinine import QuinineArgumentParser
import yaml
from schema import schema

from models import *
from trajectories import *
from kalman_filter import *
from plotting import *

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(seed=0)

def main(args):
    if args.task.matrix_dim is not None:
        A = np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/A.out", delimiter=',')
        B = np.expand_dims(np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/B.out", delimiter=','), axis=1) # Not used, should be zero
        C = np.expand_dims(np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/C.out", delimiter=','), axis=0)
        state_dim, obs_dim = A.shape[0], C.shape[0]

        noise_block = np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/noise_block.out", delimiter=',')
        Q, R, S = noise_block[0:state_dim, 0:state_dim], noise_block[state_dim:state_dim+obs_dim, state_dim:state_dim+obs_dim], noise_block[0:state_dim, state_dim:state_dim+obs_dim] # S should be zero
        x0 = rng.normal(loc=0.0, scale=1.0, size=state_dim)

    traj_lengths = [100, 200, 500]
    overall_errors_means = []

    for traj_len in traj_lengths:

        args.task.traj_len = traj_len
        state_path = os.path.join(args.training.save_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_trajlen{6}_singleSys.pt'.format(args.model.family, args.task.order_n or args.task.matrix_dim, args.model.n_embd, args.model.n_head, args.model.n_layer, args.training.lr, args.task.traj_len))
        model = build_transformer_model(args, state_dim, obs_dim)
                
        if os.path.exists(state_path):
            state = torch.load(state_path)
            model.load_state_dict(state['model_state_dict'])
            starting_step = state['train_step']

        if A is not None:
            x, y = generate_traj(args.num_tests, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)
        else:
            x = np.zeros(shape=(args.num_tests, args.task.order_n, args.task.traj_len))
            y = np.zeros(shape=(args.num_tests, args.task.order_n, args.task.traj_len))

            # Different systems for each test trajectory
            for traj_num in range(args.num_tests):
                if args.task.env_name == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args.task.order_n)
                elif args.task.env_name == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args.task.order_n)
                elif args.task.env_name == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
                x[traj_num], y[traj_num] = generate_traj(1, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)
        input_ys = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1)
        recv = model(input_ys).detach().cpu().numpy()[:, 1:, :]

        if args.model.family == 'gpt': true_values = torch.tensor(y[:, :, 1:-1]).permute(0, 2, 1).detach().cpu().numpy()

        errs = np.linalg.norm(recv - true_values, axis=2)
        overall_errors = np.linalg.norm(recv - true_values, axis=(1, 2))**2 / args.task.traj_len # overall errors for each trajectory, divided by trajectory length 

        taskname = {'gpt': 'nextstate_prediction', 'bert':'truestate_recovery'}[args.model.family] 

        for baseline_model in get_baselines(taskname, A, C, Q, R, x[:, :, 0], x[:, :, -1]):
            recv = baseline_model(input_ys.detach().cpu().numpy())
            if args.model.family == 'gpt':
                errs = np.linalg.norm(recv[:, :-1, :] - true_values, axis=2)
            if baseline_model.name == 'KalmanFilter':
                overall_kf_errors = np.linalg.norm(recv[:, :-1, :] - true_values, axis=(1, 2)) / args.task.traj_len
            
            overall_errors_means.append(overall_errors - overall_kf_errors)
    # Plot the overall errors vs trajectory lengths
    plt.figure()
    plt.xscale('log')
    plt.legend()
    plt.xlabel("Trajectory Length")
    plt.ylabel("Overall Error")
    plt.show()
    plt.savefig('./plots')

if __name__ == '__main__':
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Arguments: {args}")
    main(args)