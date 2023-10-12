import os
import torch
from torch import nn
import numpy as np
import argparse
from matplotlib import pyplot as plt

from models import *
from trajectories import *
from kalman_filter import *
from plotting import *

import wandb

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(seed=0)

def train_step(model, ys, optimizer, loss_func):
    optimizer.zero_grad()
    y_input = torch.from_numpy(ys[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(2, 0, 1) # seq_len x batch_size x obs_dim
    y_output = torch.from_numpy(ys[:, :, 1:]).to(myDevice, dtype=torch.float32).permute(2, 0, 1)
    prediction = model(y_input)
    loss = loss_func(prediction, y_output)
    loss.backward()
    # Gradient clipping, as done in https://discuss.huggingface.co/t/why-is-grad-norm-clipping-done-during-training-by-default/1866
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2) 
    optimizer.step()
    return loss.detach().item()

def train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim): # Train a model to fit a single system, characterized by A, C, Q, R, x0, state_dim, obs_dim
    model.train() # put model in training mode.
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args['lr']), 'adamw': torch.optim.AdamW(model.parameters(), lr=args['lr'])}.get(args['optim'], torch.optim.SGD(model.parameters(), lr=args['lr']))
    
    state_path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}_singleSys.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr']))
    if os.path.exists(state_path): state = torch.load(state_path); model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); return 

    loss_func = torch.nn.MSELoss()
    S = np.zeros(shape=(obs_dim, obs_dim))
    for i in range(args['num_iterations']): 
        # Generate new training data each iteration
        x, y = generate_traj(args['batch_size'], args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)
        curr_loss = train_step(model, y, optimizer, loss_func)
        wandb.log({"Iteration": i, "Loss": curr_loss})

    torch.save( {"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict()}, state_path)

def train_stable_sys(model, args): # Train a model to fit multiple stable systems, whose eigenvalues are randomly chosen in cc pairs within the unit circle
    model.train() # Put model in training mode.
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args['lr']), 'adamw': torch.optim.AdamW(model.parameters(), lr=args['lr'])}.get(args['optim'], torch.optim.SGD(model.parameters(), lr=args['lr']))
    
    state_path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}_stableSys.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr']))
    if os.path.exists(state_path): state = torch.load(state_path); model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); return
    
    loss_func = torch.nn.MSELoss()
    for i in range(args['num_iterations']):
        x, y = np.zeros(shape=(args['batch_size'], args['order_n'], args['traj_len'] + 1)), np.zeros(shape=(args['batch_size'], args['order_n'], args['traj_len'] + 1))
        S = np.zeros(shape=(args['order_n'], args['order_n']))
        # Different systems
        for traj_num in range(args['batch_size']):
            A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
            x[traj_num], y[traj_num] = generate_traj(1, args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)
        curr_loss = train_step(model, y, optimizer, loss_func)
        wandb.log({"Iteration": i, "Loss": curr_loss})

    torch.save( {"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict()}, state_path)

def train_nontrivial_sys(model, args): # Train a GPT model to fit multiple nontrivial jordan block systems, whose eigenvalues are randomly chosen on the unit circle.
    model.train()
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args['lr']), 'adamw': torch.optim.AdamW(model.parameters(), lr=args['lr'])}.get(args['optim'], torch.optim.SGD(model.parameters(), lr=args['lr']))

    state_path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}_nontrivialSys.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr']))
    if os.path.exists(state_path): state = torch.load(state_path); model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); return
    
    loss_func = torch.nn.MSELoss()
    for i in range(args['num_iterations']):
        x, y = np.zeros(shape=(args['batch_size'], args['order_n'], args['traj_len'] + 1)), np.zeros(shape=(args['batch_size'], args['order_n'], args['traj_len'] + 1))
        S = np.zeros(shape=(args['order_n'],  args['order_n']))
        # Different systems
        for traj_num in range(args['batch_size']):
            A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
            x[traj_num], y[traj_num] = generate_traj(1, args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)
        curr_loss = train_step(model, y, optimizer, loss_func)
        wandb.log({"Iteration": i, "Loss": curr_loss})

    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, state_path)

    
def visualize(trained_model, args, A=None, C=None, Q=None, R=None, x0=None, state_dim=None, obs_dim=None):
    trained_model.eval()
    S = np.zeros(shape=(args['order_n'], args['order_n']))

    if A is not None:
        x, y = generate_traj(args['num_tests'], args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)
    else:
        x, y = np.zeros(shape=(args['num_tests'], args['order_n'], args['traj_len'] + 1)), np.zeros(shape=(args['num_tests'], args['order_n'], args['traj_len'] + 1))
        print("Num tests is", args['num_tests'])
        for traj_num in range(args['num_tests']):
            if args['env_name'] == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
            elif args['env_name'] == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
            x[traj_num], y[traj_num] = generate_traj(1, args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)    

    # Plot error per timestep.
    plt.figure()
    plt.title("Error per timestep") 
    plt.xlabel("Timestep")
    plt.ylabel("Error")

    # All tensors should have shape (batch_size, seq_len, state_dim)
    input_ys = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(2, 0, 1)
    recv_gpt = trained_model(input_ys).detach().cpu().numpy()
    true_values = torch.tensor(y[:, :, 1:]).permute(2, 0, 1).detach().cpu().numpy()
    err_ys_gpt = np.linalg.norm(recv_gpt - true_values, axis=2)
    medians = np.quantile(err_ys_gpt, 0.5, axis=1)
    plt.scatter(range(len(medians)), medians, label='Median Error of GPT')

    for baseline_model in get_relevant_baselines('nextstate_prediction', A, C, Q, R, x0):
        recv_baseline = baseline_model(input_ys.detach().cpu().numpy())
        true_values = torch.tensor(y[:, :, 1:]).permute(2, 0, 1).detach().numpy()
        # import ipdb; ipdb.set_trace()
        errs = np.array([[C @ recv_baseline[i, j] - true_values[i, j] for j in range(recv_baseline.shape[1])] for i in range(recv_baseline.shape[0])])
        errs_ys_baseline = np.linalg.norm(errs, axis=1)
        medians = np.quantile(errs_ys_baseline, 0.5, axis=1)
        plt.scatter(range(len(medians)), medians, label='Median Error of {0}'.format(baseline_model.name))

    plt.legend()
    plt.show()

def main(args):
    run = wandb.init(project='dynamical-systems',
        config={'env_name': args['env_name'], 'learning rate': args['lr'], 'iterations':args['num_iterations'], 'optimizer':args['optim'], 'batchsize':args['batch_size'],
            'gpt_n_embd':args['gpt_n_embd'], 'gpt_n_layer':args['gpt_n_layer'], 'gpt_n_head':args['gpt_n_head']})
    model = GPTModel(n_dims_token=args['order_n'], n_positions=args['traj_len'], n_embd=args['gpt_n_embd'], n_layer=args['gpt_n_layer'], n_head=args['gpt_n_head'])
    if args['env_name'] == 'so2':
        A, C, Q, R, x0, state_dim, obs_dim = so2_params()
        train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim)
        visualize(model, args, A, C, Q, R, x0, state_dim, obs_dim)
    elif args['env_name'] == 'stable_sys':
        train_stable_sys(model, args)
        visualize(model, args)
    elif args['env_name'] == 'nontrivial_sys':
        train_nontrivial_sys(model, args)
        visualize(model, args)
    else: print('Invalid env_name')
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='so2', 
        choices=('stable_sys', 'nontrivial_sys', 'so2', 'so3', 'smd', 'motion', 'accel')
    )
    parser.add_argument('--num_tests', type=int, default=100)
    parser.add_argument('--traj_len', type=int, default=200)
    parser.add_argument('--order_n', type=int, default=2)

    parser.add_argument('--gpt_n_embd', type=int, default=12)
    parser.add_argument('--gpt_n_layer', type=int, default=4)
    parser.add_argument('--gpt_n_head', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--num_iterations', type=int, default=300)
    parser.add_argument('--save_path', type=str, default='./results/')

    # convert to dictionary
    params = vars(parser.parse_args())
    main(params)