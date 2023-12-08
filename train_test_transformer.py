import os
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from quinine import QuinineArgumentParser
from tqdm import tqdm
import yaml
from schema import schema

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
    y_input = torch.from_numpy(ys[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x traj_len x obs_dim
    y_output = torch.from_numpy(ys[:, :, 1:]).to(myDevice, dtype=torch.float32).permute(0, 2, 1) # batch_size x traj_len x obs_dim
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
    optimizer = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}.get(args.training.optim)(model.parameters(), lr=args.training.lr)
    
    starting_step = 0
    state_path = os.path.join(args.training.save_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_singleSys.pt'.format(args.model.family, args.task.order_n, args.model.n_embd, args.model.n_head, args.model.n_layer, args.training.lr))
    print("args", args.training.save_path)
    print("state_path", state_path)
    if os.path.exists(state_path): 
        state = torch.load(state_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        starting_step = state['train_step']

    loss_func = torch.nn.MSELoss()
    for i in tqdm(range(starting_step, args.training.num_iterations)):
        # Generate new training data each iteration
        x, y = generate_traj(args.training.batch_size, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)

        # Training Step and Logging
        if args.model.family == 'gpt': curr_loss = train_step_gpt(model, y, optimizer, loss_func)
        elif args.model.family == 'bert': curr_loss = train_step_bert(model, x[:, :, :-1], y[:, :, :-1], optimizer, loss_func)

        if i % args.wandb.log_every_steps == 0: wandb.log({'Iteration': i, 'Loss': curr_loss})
        if i % args.training.save_every_steps == 0: torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict(), 'train_step': i}, state_path)

    torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict(), "train_step": i}, state_path)
    
def train_general_system(model, args): 
    '''train_general_system(model: nn.Module, args: dict)
    Train a transformer to perform simultaneous identification and filtering of noise on a system, 
    where the system behavior is given by args.task.env_name
    '''
    optimizer = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}.get(args.training.optim, torch.optim.SGD)(model.parameters(), lr=args.training.lr)
    
    starting_step = 0
    state_path = os.path.join(args.training.save_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_singleSys.pt'.format(args.model.family, args.task.order_n, args.model.n_embd, args.model.n_head, args.model.n_layer, args.training.lr))
    if os.path.exists(state_path): 
        state = torch.load(state_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        starting_step = state['train_step']

    loss_func = torch.nn.MSELoss()
    for i in tqdm(range(starting_step, args.training.num_iterations)):
        
        # Generate batch_size trajectories that each come from different systems
        x, y = np.zeros(shape=(args.training.batch_size, args.task.order_n, args.task.traj_len)), np.zeros(shape=(args.training.batch_size, args.task.order_n, args.task.traj_len))
        for traj_num in range(args.training.batch_size):
            if args.task.env_name == 'so2': angle = np.random.random() * 90; A, C, Q, R, x0, state_dim, obs_dim = so2_params(angle)
            elif args.task.env_name == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args.task.order_n)
            elif args.task.env_name == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args.task.order_n)
            else: print('Invalid Environment Name'); return
            x[traj_num], y[traj_num] = generate_traj(1, args.task.traj_len, A, C, Q, R, x0, rng, state_dim)

        # Train Step and Logging
        if args.model.family == 'gpt': curr_loss = train_step_gpt(model, y, optimizer, loss_func)
        elif args.model.family == 'bert': curr_loss = train_step_bert(model, x, y, optimizer, loss_func)
        if i % args.wandb.log_every_steps == 0: wandb.log({'Iteration': i, 'Loss': curr_loss})
        if i % args.training.save_every_steps == 0: torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict(), 'train_step': i}, state_path)

    torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict(), 'train_step': i}, state_path)

def visualize(trained_model, args, A=None, C=None, Q=None, R=None, x0=None, state_dim=None, obs_dim=None):
    trained_model.eval()

    if A is not None:
        x, y = generate_traj(args.num_tests, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)
    else:
        x, y = np.zeros(shape=(args.num_tests, args.task.order_n, args.task.traj_len)), np.zeros(shape=(args.num_tests, args.task.order_n, args.task.traj_len))
        for traj_num in range(args.num_tests):
            if args.task.env_name == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args.task.order_n)
            elif args.task.env_name == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args.task.order_n)
            elif args.task.env_name == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
            x[traj_num], y[traj_num] = generate_traj(1, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)   

    # All tensors should have shape (batch_size, seq_len, state_dim)
    input_ys = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1)
    recv = trained_model(input_ys).detach().cpu().numpy()

    if args.model.family == 'gpt': true_values = torch.tensor(y[:, :, 1:]).permute(0, 2, 1).detach().cpu().numpy()
    elif args.model.family == 'bert': true_values = torch.tensor(x[:, :, :-1]).permute(0, 2, 1).detach().cpu().numpy()

    errs = np.linalg.norm(recv - true_values, axis=2)
    # Plot the errors over time
    metrics = {"Transformer": {'med': np.quantile(errs, 0.5, axis=0),
                              'q1': np.quantile(errs, 0.25, axis=0),
                              'q3': np.quantile(errs, 0.75, axis=0)}}
    taskname = {'gpt':'nextstate_prediction', 'bert':'truestate_recovery'}[args.model.family]

    plt.figure()
    plt.plot(*input_ys[0].T[:2], label="Ys")
    plt.plot(*torch.tensor(x).permute(0, 2, 1).detach().cpu().numpy()[0].T[:2], label="Xs")
    plt.plot(*recv[0].T[:2], label="Recovered by Transformer")
    for baseline_model in get_relevant_baselines(taskname, A, C, Q, R, x[:, :, 0], x[:, :, -1]):
        print("Model", baseline_model.name)
        recv = baseline_model(y[:, :, :-1])
        plt.plot(*recv[0], label="Recovered by {0}".format(baseline_model.name))
        errs = np.linalg.norm(recv - y[:, :, 1:], axis=1)
        metrics[baseline_model.name] = {'med': list(np.quantile(errs, 0.5, axis=0)),
                                        'q1': list(np.quantile(errs, 0.25, axis=0)),
                                        'q3': list(np.quantile(errs, 0.75, axis=0))} 
    plt.legend()
    plt.title("Trajectory and Measurements")
    plt.show()

    plot(metrics, args, title="Errors over Time for various methods")

def main(args):
    run = wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        config=args.__dict__,
        notes=args.wandb.notes
    )

    if args.task.matrix_dim is not None:
        A = np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/A.out", delimiter=',')
        print(("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else ""))
        B = np.expand_dims(np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/B.out", delimiter=','), axis=1) # Not used, should be zero
        C = np.expand_dims(np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/C.out", delimiter=','), axis=0)
        state_dim, obs_dim = A.shape[0], C.shape[0]
        
        noise_block = np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/noise_block.out", delimiter=',')
        Q, R, S = noise_block[0:state_dim, 0:state_dim], noise_block[state_dim:state_dim+obs_dim, state_dim:state_dim+obs_dim], noise_block[0:state_dim, state_dim:state_dim+obs_dim] # S should be zero
        x0 = rng.normal(loc=0.0, scale=1.0, size=state_dim)

        model = build_transformer_model(args, state_dim, obs_dim)
        model.train() # put model in training mode

        train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim)
        visualize(model, args, A, C, Q, R, x0, state_dim, obs_dim)

    else:
        if args.task.task == 'single':
            if args.task.env_name == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
            elif args.task.env_name == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args.order)
            elif args.task.env_name == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args.order)
            
            model = build_transformer_model(args, state_dim, obs_dim)
            model.train() # put model in training mode

            train_single_system(model, args, A, C, Q, R, x0, state_dim, obs_dim)
            visualize(model, args, A, C, Q, R, x0, state_dim, obs_dim)

        elif args.task.task == 'general':
            model = build_transformer_model(args, args.task.order_n, args.task.order_n)
            model.train() # put model in training mode

            train_general_system(model, args)
            visualize(model, args)

    wandb.finish()

if __name__ == '__main__':
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print(f"Arguments: {args}")
    main(args)