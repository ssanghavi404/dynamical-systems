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
    # Input and output both have shape (batch_size, seq_len, d_curr)
    y_input = torch.from_numpy(ys[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) 
    y_output = torch.from_numpy(ys[:, :, 1:]).to(myDevice, dtype=torch.float32).permute(0, 2, 1)
    pred = model(y_input)
    loss = loss_func(pred, y_output)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), pred.detach()

def train_single_system(model, args, A, C, Q, R, x0, state_dim=None, obs_dim=None):
    '''train_single_system(model: nn.Module, args: dict, A: np.array, C: np.array,
    Q: np.array, R: np.array, x0: np.array, state_dim: int, obs_dim: int)
    Train a model to fit a single system, characterized by A, C, Q, R, with starting state x0'''
    if state_dim is None: state_dim = A.shape[0]
    if obs_dim is None: obs_dim = C.shape[0]

    model.train() # put model in training mode.

    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args['lr']), 'adamw': torch.optim.AdamW(model.parameters(), lr=args['lr'])}.get(args['optim'], torch.optim.SGD(model.parameters(), lr=args['lr']))

    # Model Architecture - Load from saved path if it exists already
    state_path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}_singleSys_{5}_numIt{6}.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr'], args['env_name'], args['num_iterations']))
    if os.path.exists(state_path): state = torch.load(state_path); model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); return 
    
    # Loss Function and Training Loop
    loss_func = torch.nn.MSELoss()
    for i in range(args['num_iterations']): 
        # Generate new training data each iteration
        x, y = generate_traj(args['batch_size'], args['traj_len'], A, C, Q, R, x0, rng, state_dim, obs_dim)
        curr_loss, pred = train_step(model, y, optimizer, loss_func)
        wandb.log({"Iteration": i, "Loss": curr_loss})
    torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict()}, state_path)
    
    return model

def visualize(trained_model, args, A=None, C=None, Q=None, R=None, x0=None, state_dim=None, obs_dim=None):
    trained_model.eval()
    if A is None:
        if args['env_name'] == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
        elif args['env_name'] == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
        elif args['env_name'] == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
    x, y = generate_traj(args['num_tests'], args['traj_len'], A, C, Q, R, x0, rng, state_dim, obs_dim)

    y_torch = torch.from_numpy(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x seq_len x d_curr

    plt.figure()
    plt.plot(*x[1], label="Trajectory")
    plt.plot(*y[1], label="Measured")

    # Plot the errors over time
    metrics = {}
    recv = trained_model(y_torch).permute(0, 2, 1).detach().cpu().numpy()
    plt.plot(*recv[0], label="Recovered by Transformer")

    errs = np.linalg.norm(recv - y[:, :, 1:], axis=1)
    metrics["Transformer"] = {'med': np.quantile(errs, 0.5, axis=0), 
                              'q1': np.quantile(errs, 0.25, axis=0), 
                              'q3': np.quantile(errs, 0.75, axis=0)}
    for baseline_model in get_relevant_baselines('nextstate_prediction', A, C, Q, R, x0)[:4]:
        recv = baseline_model(y[:, :, :-1])
        errs = np.linalg.norm(recv - y[:, :, 1:], axis=1) # These "RECOVERED" Will actually be the recovered x's. The y's will be multiples of that. 
        metrics[baseline_model.name] = {'med': list(np.quantile(errs, 0.5, axis=0)),
                                        'q1': list(np.quantile(errs, 0.25, axis=0)),
                                        'q3': list(np.quantile(errs, 0.75, axis=0))}   
        plt.plot(*recv[1], label="Recovered by {0}".format(baseline_model.name))
    plt.legend()
    plt.savefig('gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}_singleSys.jpg'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr'], args['env_name'], args['num_iterations']))
    plot(metrics, title="Median Errors over Time", num_iterations=args['num_iterations'])

def main(args):
    run = wandb.init(project='dynamical-systems',
        config={'env_name': args['env_name'], 'learning rate': args['lr'], 'iterations':args['num_iterations'], 'optimizer':args['optim'], 'batchsize':args['batch_size'], 'gpt_n_embd':args['gpt_n_embd'], 'gpt_n_layer':args['gpt_n_layer'], 'gpt_n_head':args['gpt_n_head']})
    model = GPTModel(n_dims_token=args['order_n'], n_positions=args['traj_len'], n_embd=args['gpt_n_embd'], n_layer=args['gpt_n_layer'], n_head=args['gpt_n_head'])

    if args['task'] == 'single':
        if args['env_name'] == 'so2': 
            A, C, Q, R, x0, state_dim, obs_dim = so2_params()
        elif args['env_name'] == 'stable_sys':
            A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
        elif args['env_name'] == 'nontrivial_sys':
            A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
        else: print('Invalid Environment Name')
        train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim)

    elif args['task'] == 'general':
        if args['env_name'] == 'so2':
            train_so2_sys(model, args)
        elif args['env_name'] == 'stable_sys':
            train_stable_sys(model, args)
        elif args['env_name'] == 'nontrivial_sys':
            train_nontrivial_sys(model, args)
        else: print('Invalid Environment Name')
    
    visualize(model, args)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='so2', choices=('stable_sys', 'nontrivial_sys', 'so2'))

    parser.add_argument('--num_tests', type=int, default=100)
    parser.add_argument('--traj_len', type=int, default=200)
    parser.add_argument('--order_n', type=int, default=2)
    parser.add_argument('--task', type=str, default='single', choices=('single', 'general'))

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