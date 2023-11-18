import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import numpy as np
from kalman_filter import *
from helpers import *

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_relevant_baselines(task_name, A, C, Q, R, x0, xf=None):
    task_to_baselines = {
        'nextstate_prediction': [
            (ZeroPredictorModel, {}),
            (PrevPredictorModel, {}),
            (KfModel, {'A':A, 'C':C, 'Q':Q, 'R':R, 'x0':x0}),
            (IdKfModel, {'Q':Q, 'R':R, 'x0':x0})
        ],  
        'truestates_recovery': [
            (ZeroPredictorModel, {}),
            (LerpModel, {}),
            # (KfSmoothedModel, {'A':A, 'Q':Q, 'R':R, 'x0':x0, 'xf':xf}),
            (LSOptModel, {'A':A, 'C':C, 'Q':Q, 'R':R}),
        ]
    }
    models = []
    for model_cls, kwargs in task_to_baselines[task_name]:
        models.append(model_cls(**kwargs))
    return models

class ZeroPredictorModel:
    # Predict 0 at each timestep
    def __init__(self):
        self.name = 'ZeroPredictor'
    def __call__(self, ys):
        return np.zeros_like(ys)
    
class PrevPredictorModel:
    # Predict the previous token at each timestep
    def __init__(self):
        self.name = 'PrevPredictor'
    def __call__(self, ys):
        return ys

class KfModel:
    # Perform Kalman Filtering and use the receovered states at each timetep, with the known A, C, Q, R matrices
    def __init__(self, A, C, Q, R, x0):
        self.name = 'KalmanFilter'
        self.A, self.C, self.Q, self.R, self.x0 = A, C, Q, R, x0
    def __call__(self, ys):
        recv = np.zeros_like(ys)
        num_traj, obs_dim, traj_len = ys.shape
        for trajNum in range(num_traj):
            kinematics = KFilter(self.A, self.C, self.Q, self.R, state=self.x0)
            recv[trajNum, :, :] = kinematics.simulate(ys[trajNum, :, :])
        return recv

class IdKfModel:
    def __init__(self, Q, R, x0=None): 
        self.name = 'SystemIdKalmanFilter'
        self.Q = Q
        self.R = R
        self.x0 = x0

    def __call__(self,  ys):
        num_traj, obs_dim, traj_len = ys.shape
        recv = np.zeros_like(ys)
        for i in range(num_traj):
            # No peeking on what is the actual A matrix
            # C is assumed to be the identity matrix
            A_unk = np.zeros(shape=(obs_dim, obs_dim))
            C = np.eye(obs_dim, obs_dim)
            kinematics = KFilter(A_unk, C, self.Q, self.R, state=self.x0)
            for t in range(traj_len):
                A_found = system_id(ys[i, :, :].T, t, self.x0)
                kinematics.A = A_found
                kinematics.predict()
                kinematics.update(ys[i, :, t])
                recv[i, :, t] = kinematics.state
        return recv

class LerpModel:
    def __init__(self):
        self.name = 'LinearInterpolation'

    def __call__(self, ys):
        traj_len, obs_dim, num_traj = ys.shape
        recv = np.zeros_like(ys)
        recv[0, :, :] = ys[0, :, :]
        for t in range(1, traj_len - 1):
            recv[t, :, :] =(ys[t - 1, :, :] + ys[t + 1, :, :]) / 2
        recv[traj_len-1, :, :] = ys[traj_len-1, :, :]
        return recv

# TODO - call Wentinn's code here. For now, it's just returning the previous ys.
class LearnedKfModel:
    def __init__(self):
        self.name = 'LearnedKalmanFilter'
    
    def __call__(self, ys):
        return ys
        
class KfSmoothedModel:
    def __init__(self, A, Q, R, x0, xf):
        self.name = "KfSmoothed"
        C = np.eye(*A.shape)
        self.forward = KFilter(A, C, Q, R, x0)
        self.backward = KFilter(np.linalg.inv(A), C, Q, R, xf)

    def __call__(self, ys):
        traj_len, num_traj, obs_dim = ys.shape
        recv = np.zeros_like(ys)
        for trajNum in range(num_traj):
            fltr_fwd = self.forward.simulate(ys[:, trajNum, :].T).T
            flipped_ys = np.flip(ys[:, trajNum, :], axis=0)
            fltr_bkwd = self.backward.simulate(flipped_ys.T).T
            fltr_bkwd = np.flip(fltr_bkwd, axis=0)
            recv[:, trajNum, :] = (fltr_fwd + fltr_bkwd) / 2
        return recv

class LSOptModel:
    def __init__(self, A, C, Q, R):
        self.name = "LeastSquaresOpt"
        self.A, self.C, self.Q, self.R = A, C, Q, R
        
    def __call__(self, ys):
        traj_len, num_traj, obs_dim = ys.shape
        print("ys has shape", ys.shape)
        recv = np.zeros_like(ys)
        for traj_num in range(num_traj):
            recv[:, traj_num, :] = optimal_traj(self.A, self.C, self.Q, self.R, ys[:, traj_num, :])
        return recv

class GPTModel(nn.Module):
    def __init__(self, n_dims_token, n_positions, n_embd, n_layer, n_head):
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
        self.n_positions = n_positions
        self.n_dims_token = n_dims_token
        self.name = "GPTModel"
        self._read_in = nn.Linear(n_dims_token, n_embd).to(myDevice, dtype=torch.float32)
        self._backbone = GPT2Model(configuration).to(myDevice, dtype=torch.float32)
        self._read_out = nn.Linear(n_embd, n_dims_token).to(myDevice, dtype=torch.float32)

    def forward(self, ys):
        embeds = self._read_in(ys)
        output = self._backbone(inputs_embeds=embeds)
        prediction = self._read_out(output.last_hidden_state)
        return prediction
    
# class BERTModel(nn.Module):
    
#     def __init__(self, n_dims_token, n_positions, n_embd, n_layer, n_head):
#         super(BERTModel, self).__init__()
#         self.name = "BERTModel"
#         self._read_in = nn.Linear(n_dims_token, n_embd).to(myDevice, dtype=torch.float32)
#         self._backbone = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=n_embd, dropout=0, activation="relu"), 
#             num_layers=n_layer, norm=nn.LayerNorm(n_embd)
#         )
#         self._read_out = nn.Linear(n_embd, n_dims_token).to(myDevice, dtype=torch.float32)

#     def forward(self, ys):
#         embeds = self._read_in(ys)
#         output = self._backbone(embeds)
#         prediction = self._read_out(output)
#         return prediction