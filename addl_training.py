import os
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import unittest

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

def train_step_gpt(model, ys, optimizer, criterion):
    optimizer.zero_grad()
    y_input = torch.from_numpy(np.array(ys[:, :-1, :], copy=True)).to(myDevice, dtype=torch.float32) # batch_size x traj_len x obs_dim
    y_output = torch.from_numpy(np.array(ys[:, 1:, :], copy=True)).to(myDevice, dtype=torch.float32) # batch_size x traj_len x obs_dim
    pred = model(y_input)
    loss = criterion(pred, y_output)
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train_single_system(model, optimizer, args, A, C, Q, R, x0, state_dim, obs_dim, starting_step=0):
    '''train_single_systetm(model: nn.Module, args: dict, A: np.array, C: np.array, 
            Q: np.array, R: np.array, x0: np.array, state_dim: int, obs_dim: int, starting_step: int)
    Train a model to fit a single system, characterized by A, C, Q, R, with starting state x0'''
    state_path = os.path.join(args.training.save_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_trajlen{6}.pt'.format(args.model.family, args.task.order_n or args.task.matrix_dim, args.model.n_embd, args.model.n_head, args.model.n_layer, args.training.lr, args.training.traj_len))

    criterion = torch.nn.MSELoss()
    for i in tqdm(range(starting_step, args.training.num_iterations)):
        # Generate new training data each iteration
        x, y = generate_traj(args.training.batch_size, args.training.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)

        # Training Step and Logging
        if args.model.family == 'gpt': curr_loss = train_step_gpt(model, y, optimizer, criterion)

        if i % args.wandb.log_every_steps == 0: wandb.log({'Iteration': i, 'Loss': curr_loss})
        if i % args.training.save_every_steps == 0: torch.save({"model_state_dict": model.state_dict(),  "optimizer_state_dict": optimizer.state_dict(), 'train_step': i}, state_path + "chpt{0}".format(i))

def evaluate_nextstate_model(model, y):
    '''evaluate_model(model: BaseModel, y: np.array)
    Evaluate a trained model on new traces of data'''
    pred = model(y[:, :-1, :])
    errors = [np.mean((pred[i] - y[i, 1:, :])**2) for i in range(y.shape[0])]
    return errors

def load_model_and_optimizer(state_path, model, optimizer):
    '''load_model(state_path: str, model: BaseModel, optimizer: torch.optim) --> int starting step
    Loads the model from state_path and returns the time that the starting step for training
    '''
    if os.path.exists(state_path): 
        state = torch.load(state_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
    else:
        print("No model to load")

def main(args):

    if args.task.matrix_dim is not None:
        A = np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/A.out", delimiter=',')
        B = np.expand_dims(np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/B.out", delimiter=','), axis=1) # Not used, should be zero
        C = np.expand_dims(np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/C.out", delimiter=','), axis=0)
        state_dim, obs_dim = A.shape[0], C.shape[0]
        
        noise_block = np.loadtxt("./matrices/" + str(args.task.matrix_dim) + "dim_scalar_system_matrices" + ("_" + str(args.task.matrix_type) if args.task.matrix_type is not None else "") + "/noise_block.out", delimiter=',')
        Q, R, S = noise_block[0:state_dim, 0:state_dim], noise_block[state_dim:state_dim+obs_dim, state_dim:state_dim+obs_dim], noise_block[0:state_dim, state_dim:state_dim+obs_dim] # S should be zero
        x0 = rng.normal(loc=0.0, scale=1.0, size=state_dim)

        transformer_model = build_transformer_model(args, state_dim, obs_dim)
        state_path = os.path.join(args.training.save_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr0.0006_trajlen{6}.ptchpt280'.format(args.model.family, args.task.order_n or args.task.matrix_dim, args.model.n_embd, args.model.n_head, args.model.n_layer, args.training.lr, args.training.traj_len))
        optimizer = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}.get(args.training.optim)(transformer_model.parameters(), lr=args.training.lr)

        print("state_path", state_path)
        load_model_and_optimizer(state_path, transformer_model, optimizer)

        run = wandb.init(project=args.wandb.project, entity=args.wandb.entity, config=args.__dict__, notes=args.wandb.notes)
        transformer_model.train() # put model in training mode
        train_single_system(transformer_model, optimizer, args, A, C, Q, R, x0, state_dim, obs_dim, starting_step=280)
        torch.save({"model_state_dict": transformer_model.state_dict(),  "optimizer_state_dict": optimizer.state_dict()}, state_path)
        wandb.finish()

        transformer_model.eval()
        x, y = generate_traj(args.testing.num_tests, args.testing.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)
        start_states = np.zeros(shape=(args.testing.num_tests, state_dim))
        for i in range(args.testing.num_tests): start_states[i] = x0

        baseline = KfModel(A=A, C=C, Q=Q, R=R, start_states=start_states)
        errors_transformer = evaluate_nextstate_model(transformer_model, y)
        errors_baseline = evaluate_nextstate_model(baseline, y)

        gap_values = [errors_transformer[i] - errors_baseline[i] for i in range(args.testing.num_tests)]
        
        N = len(gap_values)
        s = np.sort(gap_values)
        f = np.array(range(N))/N
        plt.figure()
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(s, f, label='cdf')
        plt.plot(s, np.ones_like(f) - f, label='ccdf')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    main(args)