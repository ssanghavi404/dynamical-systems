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

def train_step_gpt(model, ys, optimizer, loss_func):
    optimizer.zero_grad()
    # Input and output both have shape (batch_size, seq_len, d_curr)
    y_input = torch.from_numpy(ys[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x seq_len x obs_dim
    y_output = torch.from_numpy(ys[:, :, 1:]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x seq_len x obs_dim
    pred = model(y_input)
    loss = loss_func(pred, y_output)
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train_step_bert(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    ys = torch.from_numpy(ys[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x seq_len x obs_dim
    xs = torch.from_numpy(xs[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x seq_len x obs_dim
    pred = model(ys)
    loss = loss_func(pred, xs)
    loss.backward()
    # Gradient clipping, as done in https://discuss.huggingface.co/t/why-is-grad-norm-clipping-done-during-training-by-default/1866
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2) 
    optimizer.step()
    return loss.detach().item()

def train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim):
    '''train_single_system(model: nn.Module, args: dict, A: np.array, C: np.array, 
            Q: np.array, R: np.array, x0: np.array, state_dim: int, obs_dim: int)
    Train a model to fit a single system, characterized by A, C, Q, R, with starting state x0'''
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args['lr']), 'adamw': torch.optim.AdamW(model.parameters(), lr=args['lr'])}.get(args['optim'], torch.optim.SGD(model.parameters(), lr=args['lr']))
    
    state_path = os.path.join(args['save_path'], '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_singleSys.pt'.format(args['transformer_type'], args['order_n'], args['transformer_n_embd'], args['transformer_n_head'], args['transformer_n_layer'], args['lr']))
    if os.path.exists(state_path): state = torch.load(state_path); model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); return 

    loss_func = torch.nn.MSELoss()
    for i in range(args['num_iterations']):
        # Generate new training data each iteration
        x, y = generate_traj(args['batch_size'], args['traj_len'], A, C, Q, R, x0, rng, state_dim, obs_dim)

        # Training Step and Logging
        if args['transformer_type'] == 'gpt': 
            curr_loss = train_step_gpt(model, y, optimizer, loss_func)
        elif args['transformer_type'] == 'bert': 
            curr_loss = train_step_bert(model, x[:, :, :-1], y[:, :, :-1], optimizer, loss_func)
        else: print("Invalid Transformer Type")
        wandb.log({"Iteration": i, "Loss": curr_loss})

    torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict()}, state_path)
    
def train_general_system(model, args): 
    '''train_general_system(model: nn.Module, args: dict)
    Train a transformer to perform simultaneous identification and filtering of noise on a system, 
    where the system behavior is given by args['env_name']
    '''
    optimizer = {'adam': torch.optim.Adam(model.parameters(), lr=args['lr']), 'adamw': torch.optim.AdamW(model.parameters(), lr=args['lr'])}.get(args['optim'], torch.optim.SGD(model.parameters(), lr=args['lr']))
    
    state_path = os.path.join(args['save_path'], '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_singleSys.pt'.format(args['transformer_type'], args['order_n'], args['transformer_n_embd'], args['transformer_n_head'], args['transformer_n_layer'], args['lr']))
    if os.path.exists(state_path): state = torch.load(state_path); model.load_state_dict(state['model_state_dict']); optimizer.load_state_dict(state['optimizer_state_dict']); return 

    loss_func = torch.nn.MSELoss()
    for i in range(args['num_iterations']):
        
        # Generate batch_size trajectories that each come from different systems
        x, y = np.zeros(shape=(args['batch_size'], args['state_dim'], args['traj_len'])), np.zeros(shape=(args['batch_size'], args['input_dim'], args['traj_len']))
        for traj_num in range(args['batch_size']):
            if args['env_name'] == 'so2': angle = np.random.random() * 90; A, C, Q, R, x0, state_dim, obs_dim = so2_params(angle)
            elif args['env_name'] == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
            elif args['env_name'] == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
            else: print('Invalid Environment Name'); return
            x[traj_num], y[traj_num] = generate_traj(1, args['traj_len'], A, C, Q, R, x0, rng, state_dim)

        # Train Step and Logging
        if args['transformer_type'] == 'gpt': curr_loss = train_step_gpt(model, y, optimizer, loss_func)
        elif args['transformer_type'] == 'bert': curr_loss = train_step_bert(model, x, y, optimizer, loss_func)
        else: print("Invalid Transformer Type"); return
        wandb.log({'Iteration': i, 'Loss': curr_loss})

    torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict()}, state_path)
    return model

def visualize(trained_model, args, A=None, C=None, Q=None, R=None, x0=None, state_dim=None, obs_dim=None):
    trained_model.eval()

    if A is not None:
        x, y = generate_traj(args['num_tests'], args['traj_len'], A, C, Q, R, x0, rng, state_dim, obs_dim)
    else:
        x, y = np.zeros(shape=(args['num_tests'], args['order_n'], args['traj_len'])), np.zeros(shape=(args['num_tests'], args['order_n'], args['traj_len']))
        for traj_num in range(args['num_tests']):
            if args['env_name'] == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
            elif args['env_name'] == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
            elif args['env_name'] == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
            x[traj_num], y[traj_num] = generate_traj(1, args['traj_len'], A, C, Q, R, x0, rng, state_dim, obs_dim)   

    # All tensors should have shape (batch_size, seq_len, state_dim)
    input_ys = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1)
    print("input_ys", input_ys.shape)
    recv = trained_model(input_ys).detach().cpu().numpy()
    print("Recv", recv.shape)

    if args['transformer_type'] == 'gpt':
        true_values = torch.tensor(y[:, :, 1:]).permute(0, 2, 1).detach().cpu().numpy()
    elif args['transformer_type'] == 'bert':
        true_values = torch.tensor(x[:, :, :-1]).permute(0, 2, 1).detach().cpu().numpy()

    errs = np.linalg.norm(recv - true_values, axis=2)
    print('errs', errs.shape)
    # Plot the errors over time
    metrics = {}
    metrics["Transformer"] = {'med': np.quantile(errs, 0.5, axis=0),
                              'q1': np.quantile(errs, 0.25, axis=0),
                              'q3': np.quantile(errs, 0.75, axis=0)}
    taskname = {'gpt':'nextstate_prediction', 'bert':'truestate_recovery'}[args['transformer_type']]

    plt.figure()
    # import ipdb; ipdb.set_trace()
    plt.plot(*input_ys[0].T, label="Ys")
    plt.plot(*true_values[0].T, label="Xs")
    plt.plot(*recv[0].T, label="Recovered by Transformer")
    for baseline_model in get_relevant_baselines(taskname, A, C, Q, R, x[:, :, 0], x[:, :, -1]):
        recv = baseline_model(y[:, :, :-1])
        plt.plot(*recv[0], label="Recovered by {0}".format(baseline_model.name))
        errs = np.linalg.norm(recv - y[:, :, 1:], axis=1)
        metrics[baseline_model.name] = {'med': list(np.quantile(errs, 0.5, axis=0)),
                                        'q1': list(np.quantile(errs, 0.25, axis=0)),
                                        'q3': list(np.quantile(errs, 0.75, axis=0))}  
    plt.legend()
    plt.title("Trajectory and measurements")
    plt.show()

    plot(metrics, args, title="Errors over Time for various methods")

def main(args):
    run = wandb.init(project='dynamical-systems',
        config={'env_name': args['env_name'], 'learning rate': args['lr'], 'iterations':args['num_iterations'], 'optimizer':args['optim'], 'batchsize':args['batch_size'],
            'transformer_n_embd':args['transformer_n_embd'], 'transformer_n_layer':args['transformer_n_layer'], 'transformer_n_head':args['transformer_n_head']})
    if args['transformer_type'] == 'gpt':
        model = GPTModel(n_dims_token=args['order_n'], n_positions=args['traj_len'], n_embd=args['transformer_n_embd'], n_layer=args['transformer_n_layer'], n_head=args['transformer_n_head'])
    elif args['transformer_type'] == 'bert':
        model = BERTModel(n_dims_token=args['order_n'], n_positions=args['traj_len'], n_embd=args['transformer_n_embd'], n_layer=args['transformer_n_layer'], n_head=args['transformer_n_head'])
    else: print("Invalid Model Type")

    model.train() # put model in training mode
    
    if args['task'] == 'single':
        if args['env_name'] == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
        elif args['env_name'] == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args['order_n'])
        elif args['env_name'] == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args['order_n'])
        else: print('Invalid Environment Name')
        # train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim)
        visualize(model, args, A, C, Q, R, x0, state_dim, obs_dim)
    elif args['task'] == 'general':
        train_general_system(model, args)
        visualize(model, args)
    else: print('Invalid env_name')
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='so2', choices=('stable_sys', 'nontrivial_sys', 'so2'))

    parser.add_argument('--num_tests', type=int, default=100)
    parser.add_argument('--traj_len', type=int, default=200)
    parser.add_argument('--order_n', type=int, default=2)
    parser.add_argument('--task', type=str, default='single', choices=('single', 'general'))

    parser.add_argument('--transformer_type', type=str, default='gpt', choices=('gpt', 'bert'))
    parser.add_argument('--transformer_n_embd', type=int, default=32)
    parser.add_argument('--transformer_n_layer', type=int, default=3)
    parser.add_argument('--transformer_n_head', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optim', type=str, default='adamw')
    
    parser.add_argument('--num_iterations', type=int, default=300)
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--graph_path', type=str, default='./plots/')

    # convert to dictionary
    params = vars(parser.parse_args())
    main(params)