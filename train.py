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

CUT = 5

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

def train(model, args, A, C, Q, R, S, x0, state_dim, obs_dim):
    model.train() # put model in training mode.

    if args['optim'] == 'adam': optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    elif args['optim'] == 'adamw': optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])
    else: optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])

    state_path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr']))
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        return 

    loss_func = torch.nn.MSELoss()
    losses = []
    for i in range(args['num_iterations']): 
        # Generate new training data each iteration
        x, y = generate_traj(args['batch_size'], args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)
        curr_loss = train_step(model, y, optimizer, loss_func)
        wandb.log({"Iteration": i, "Loss": curr_loss})
        losses.append(curr_loss)

    train_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(train_state, state_path)

    # Plot the losses
    plt.figure()
    plt.xscale('log')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.ylabel('Loss value')
    plt.plot(losses)
    plt.show()

def visualize(trained_model, args, A, C, Q, R, S, x0, state_dim, obs_dim):
    trained_model.eval()
    path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr']))
    torch.load(path)
    x, y = generate_traj(100, args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, obs_dim)
    
    # Plot error per timestep.
    plt.figure()
    plt.title("Error per timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Error")

    # All tensors should have shape (batch_size, seq_len)
    input_ys = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(2, 0, 1)
    recv_gpt = trained_model(input_ys).detach().cpu().numpy()
    true_values = torch.tensor(y[:, :, 1:]).permute(2, 0, 1).detach().cpu().numpy()
    err_ys_gpt = np.linalg.norm(recv_gpt - true_values, axis=2)
    medians = np.quantile(err_ys_gpt, 0.5, axis=1)
    plt.scatter(range(len(medians)), medians, label='Median Error of GPT')

    for baseline_model in get_relevant_baselines('nextstate_prediction', A, C, Q, R, x0):
        print("baseline Method", baseline_model.name)
        recv_baseline = baseline_model(input_ys.detach().cpu().numpy())
        true_values = torch.tensor(y[:, :, 1:]).permute(2, 0, 1).detach().numpy()
        errs_ys_baseline = np.linalg.norm(recv_baseline - true_values, axis=2)
        medians = np.quantile(errs_ys_baseline, 0.5, axis=1)
        plt.scatter(range(len(medians)), medians, label='Median Error of {0}'.format(baseline_model.name))
        
    plt.legend()
    plt.show()

def main(args):
    A, C, Q, R, S, x0, state_dim, obs_dim = so2_params()
    run = wandb.init(
        project='dynamical-systems',
        config={
            'learning rate': args['lr'], 'iterations':args['num_iterations'], 'optimizer':args['optim'], 'batchsize':args['batch_size'],
            'gpt_n_embd':args['gpt_n_embd'], 'gpt_n_layer':args['gpt_n_layer'], 'gpt_n_head':args['gpt_n_head'],
    })
    model = GPTModel(n_dims_token=obs_dim, n_positions=args['traj_len'], n_embd=args['gpt_n_embd'], n_layer=args['gpt_n_layer'], n_head=args['gpt_n_head'])
    train(model, args, A, C, Q, R, S, x0, state_dim, obs_dim)
    visualize(model, args, A, C, Q, R, S, x0, state_dim, obs_dim)
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='stable_sys', 
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