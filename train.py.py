import os
import torch
from torch import nn
import numpy as np
import argparse
from matplotlib import pyplot as plt

from models import GPTModel
from trajectories import *
from kalman_filter import *


myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(seed=0)

def train_step(model, ys, optimizer, loss_func):
    optimizer.zero_grad()
    prediction = model(y[:-1])
    loss = loss_func(prediction, y[1:])
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
        
        # Cast to PyTorch tensors
        y_input = torch.from_numpy(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(2, 0, 1) # seq_len x batch_size x obs_dim
        y_output = torch.from_numpy(y[:, :, 1:]).to(myDevice, dtype=torch.float32).permute(2, 0, 1)
        curr_loss = train_step(model, y_input, y_output, optimizer, loss_func)
        print("Iteration", i, "curr_loss", curr_loss)
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


def visualize(model, args, A, C, Q, R, S, x0, state_dim, obs_dim):
    model.eval()
    path = os.path.join(args['save_path'], 'gpt_order{0}_emb{1}_heads{2}_layers{3}_lr{4}.pt'.format(args['order_n'], args['gpt_n_embd'], args['gpt_n_head'], args['gpt_n_layer'], args['lr']))
    torch.load(path)

    x, y = generate_traj(1, args['traj_len'], A, C, Q, R, S, x0, rng, state_dim, input_dim, obs_dim)
    
    # All tensors should have shape (batch_size, input_dim, seq_len)
    
    transformer_input = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(2, 0, 1)
    recv = model(transformer_input).detach().numpy()
    err_ys = np.linalg.norm(recv - y[:, :, 1:].T, axis=(1, 2))
    
    # Plot trajectory, measured, and recovered in space.
    plot({"Trajectory": x[0, :, 1:].T, "Measured":y[0, :, 1:].T, "Filtered":recv[:, 0, :]})
    # Plot error per timestep.
    plt.figure()
    plt.title("Error per timestep")
    plt.yscale('log')
    plt.xlabel("Timestep")
    plt.ylabel("Error")
    plt.plot(err_ys)
    plt.show()

def main(args):
    A, C, Q, R, S, x0, state_dim, obs_dim = so2_params()
    model = GPTModel(n_dims_token=obs_dim, n_positions=args['traj_len'], n_embd=args['gpt_n_embd'], n_layer=args['gpt_n_layer'], n_head=args['gpt_n_head'])
    train(model, args, A, C, Q, R, S, x0, state_dim, obs_dim)
    visualize(model, args, A, C, Q, R, S, x0, state_dim, obs_dim)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='stable_sys', 
        choices=('stable_sys', 'nontrivial_sys', 'so2', 'so3', 'smd', 'motion', 'accel')
    )
    parser.add_argument('--num_tests', type=int, default=100)
    parser.add_argument('--traj_len', type=int, default=150)
    parser.add_argument('--order_n', type=int, default=2)

    parser.add_argument('--gpt_n_embd', type=int, default=12)
    parser.add_argument('--gpt_n_layer', type=int, default=4)
    parser.add_argument('--gpt_n_head', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='./results/')

    # convert to dictionary
    params = vars(parser.parse_args())
    main(params)