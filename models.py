import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import numpy as np
from kalman_filter import *
from helpers import *
from abc import ABC

import wandb

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_transformer_model(config, state_dim, obs_dim):
    if config.model.family == "gpt":
        model = GPTModel(
            n_dims_obs=obs_dim,
            n_positions=config.model.n_positions,
            n_embd=config.model.n_embd,
            n_layer=config.model.n_layer,
            n_head=config.model.n_head,
        )
    return model


def get_baselines(task_name, A, C, Q, R, start_states, end_states=None):
    if task_name == "nextstate_prediction":
        return [
            # ZeroPredictorModel(pred_dim=C.shape[0]),
            # PrevPredictorModel(),
            KfModel(A=A, C=C, Q=Q, R=R, start_states=start_states),
            # IdKfModel(Q=Q, R=R, start_states=start_states),
        ]
    else:
        print("Invalid tasknsame")
        return []


class BaseModel(ABC):
    def __init__(self):
        pass

    def __call__(self):
        pass


class ZeroPredictorModel(BaseModel):
    # Predict 0 at each timestep
    def __init__(self, pred_dim):
        self.name = "ZeroPredictor"
        self.pred_dim = (
            pred_dim  # size of the predictions (obs_dim for GPT, state_dim for BERT)
        )

    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        return np.zeros(shape=(num_traj, traj_len, self.pred_dim))


class PrevPredictorModel(BaseModel):
    # Predict the previous token at each timestep. basl
    def __init__(self):
        self.name = "PrevPredictor"

    def __call__(self, ys):
        return ys


class KfModel(BaseModel):
    # Perform Kalman Filtering to recover states at each timetep, with the known A, C, Q, R matrices
    # Return the observations of the states that we have recovered. 
    def __init__(self, A, C, Q, R, start_states):
        self.name = "KalmanFilter"
        self.A, self.C, self.Q, self.R, self.start_states = A, C, Q, R, start_states

    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        state_dim = self.A.shape[0]
        recv = np.zeros_like(ys)  # Recovered Ys
        for trajNum in range(num_traj):
            kinematics = KFilter(
                self.A, self.C, self.Q, self.R, state=np.zeros(state_dim)#state=self.start_states[trajNum]
            )
            kinematics.run_till_ss()
            recv_xs = kinematics.simulate(ys[trajNum, :, :].T).T
            recv[trajNum, :-1, :] = np.array(
                [[*(self.C @ recv_xs[i, :])] for i in range(1, traj_len)]
            )
            kinematics.predict()  # Last step ahead
            recv[trajNum, -1, :] = self.C @ kinematics.state
        return recv

class GPTModel(nn.Module, BaseModel):
    def __init__(self, n_dims_obs, n_positions, n_embd, n_layer, n_head):
        super(GPTModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = "GPTModel"
        self.n_dims_obs = n_dims_obs
        self._read_in = nn.Linear(n_dims_obs, n_embd).to(myDevice, dtype=torch.float32)
        self._backbone = GPT2Model(configuration).to(myDevice, dtype=torch.float32)
        self._read_out = nn.Linear(n_embd, n_dims_obs).to(myDevice, dtype=torch.float32)

    def forward(self, ys):
        # During training, we will be providing the PyTorch tensors with the necessary off-by-one so that we can pass gradients through.
        if self.training:
            embeds = self._read_in(ys)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state 
            prediction = self._read_out(output)
            return prediction
        # However, during evaluation, we will be providing a numpy tensor of the ys, without performing the swapaxes as needed
        else:
            y_input = torch.from_numpy(ys).to(myDevice, dtype=torch.float32) # batch_size x traj_len x obs_dim
            embeds = self._read_in(y_input)
            output = self._backbone(inputs_embeds=embeds).last_hidden_state
            prediction = self._read_out(output)
            return prediction.detach().cpu().numpy()
