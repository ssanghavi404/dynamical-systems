import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
import numpy as np
from kalman_filter import *
from helpers import *

import wandb

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_transformer_model(config, state_dim, obs_dim):
    if config.model.family == 'gpt':
        model = GPTModel(
            n_dims_obs=obs_dim, 
            n_positions=config.task.traj_len, 
            n_embd=config.model.n_embd, 
            n_layer=config.model.n_layer, 
            n_head=config.model.n_head)
    elif config.model.family == 'bert':
        model = BERTModel(
            n_dims_obs=obs_dim, 
            n_dims_state=state_dim, 
            n_positions=config.task.traj_len, 
            n_embd=config.model.n_embd, 
            n_layer=config.model.n_layer, 
            n_head=config.model.n_head)
    else: 
        raise NotImplementedError
    return model

def get_baselines(task_name, A, C, Q, R, start_states, end_states=None):
    if task_name == 'nextstate_prediction':
        return [
            ZeroPredictorModel(pred_dim=C.shape[0]),
            PrevPredictorModel(),
            KfModel(A=A, C=C, Q=Q, R=R, start_states=start_states),
            # IdKfModel(Q=Q, R=R, start_states=start_states),
        ]
    elif task_name == 'truestate_recovery':
        return [
            ZeroPredictorModel(pred_dim=A.shape[0]),
            LerpModel(state_dim=A.shape[0]),
            KfSmoothedModel(A=A, C=C, Q=Q, R=R, start_states=start_states, end_states=end_states),
            LSOptModel(A=A, C=C, Q=Q, R=R),
        ]
    else:
        print('Invalid taskname')
        return []
    
class ZeroPredictorModel:
    # Predict 0 at each timestep
    def __init__(self, pred_dim):
        self.name = 'ZeroPredictor'
        self.pred_dim = pred_dim # size of the predictions (obs_dim for GPT, state_dim for BERT)

    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        return np.zeros(shape=(num_traj, traj_len, self.pred_dim))
    
class PrevPredictorModel:
    # Predict the previous token at each timestep. basl
    def __init__(self):
        self.name = 'PrevPredictor'

    def __call__(self, ys):
        return ys

class KfModel:
    # Perform Kalman Filtering and use the recovered states at each timetep, with the known A, C, Q, R matrices
    def __init__(self, A, C, Q, R, start_states):
        self.name = 'KalmanFilter'
        self.A, self.C, self.Q, self.R, self.start_states = A, C, Q, R, start_states
    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        state_dim = self.A.shape[0]
        recv = np.zeros_like(ys) # Recovered Ys
        for trajNum in range(num_traj):
            kinematics = KFilter(self.A, self.C, self.Q, self.R, state=self.start_states[trajNum])
            recv_xs = kinematics.simulate(ys[trajNum, :, :].T)
            print("Recv_xs.shape", recv_xs.shape)
            print("Recv_xs", recv_xs)
            print('recv.shape', recv.shape)
            print("self.C", self.C)
            print("start", recv_xs[:, 0])
            print("first", self.C @ recv_xs[:, 0])
            print("second", self.C @ recv_xs[:, 1])
            
            print("thingToAssign", np.array([[*(self.C @ recv_xs[:, i])] for i in range(1, traj_len)]))
            recv[trajNum, :-1, :] = np.array([[*(self.C @ recv_xs[:, i])] for i in range(1, traj_len)])
            print("penultimate state", kinematics.state)
            print("penultimate obs", self.C @ kinematics.state)
            kinematics.predict() # Last step ahead 
            print("last state", kinematics.state)
            print("last obs", self.C @ kinematics.state)
            recv[trajNum, -1, :] = self.C @ kinematics.state
        return recv

class IdKfModel:
    def __init__(self, Q, R, start_states=None): 
        self.name = 'SystemIdKalmanFilter'
        self.Q = Q
        self.R = R
        self.start_states = start_states

    def __call__(self,  ys):
        num_traj, traj_len, obs_dim = ys.shape
        recv = np.zeros_like(ys)
        for traj_num in range(num_traj):
            # No peeking on what is the actual matrix
            # C is assumed to be the identity matrix
            A_unk = np.zeros(shape=(obs_dim, obs_dim))
            C = np.eye(obs_dim, obs_dim)
            kinematics = KFilter(A_unk, C, self.Q, self.R, state=self.start_states[traj_num])
            for t in range(traj_len):
                A_found = system_id(ys[traj_num, :, :].T, t, self.start_states[traj_num])
                import ipdb; ipdb.set_trace()
                kinematics.A = A_found
                kinematics.predict()
                kinematics.update(ys[traj_num, :, t])
                recv[traj_num, :, t] = C @ kinematics.state
        return recv

class LerpModel: # Note - requires that C = I
    def __init__(self, state_dim):
        self.name = 'LinearInterpolation'
        self.state_dim = state_dim

    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        recv = np.zeros(shape=(num_traj, traj_len, obs_dim)) 
        recv[:, :, 0] = ys[:, :, 0]
        for t in range(1, traj_len - 1):
            recv[:, :, t] = (ys[:, :, t - 1] + ys[:, :, t + 1]) / 2
        recv[:, :, -1] = ys[:, :, -1]
        return recv

# # TODO - call Wentinn's code here. For now, it's just returning the previous ys.
# class LearnedKfModel:
#     def __init__(self):
#         self.name = 'LearnedKalmanFilter'
    
#     def __call__(self, ys):
#         return ys
        
class KfSmoothedModel:
    def __init__(self, A, C, Q, R, start_states, end_states):
        self.name = "KfSmoothed"
        C = np.eye(*A.shape)
        self.A, self.C, self.Q, self.R, self.start_states, self.end_states = A, C, Q, R, start_states, end_states

    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        state_dim = self.A.shape[0]
        recv = np.zeros(shape=(num_traj, traj_len, state_dim))
        for trajNum in range(num_traj):
            forward = KFilter(self.A, self.C, self.Q, self.R, self.start_states[trajNum])
            backward = KFilter(np.linalg.inv(self.A), self.C, self.Q, self.R, self.end_states[trajNum])
            fltr_fwd = forward.simulate(ys[trajNum, :, :])
            flipped_ys = np.flip(ys[trajNum, :, :], axis=0)
            fltr_bkwd = backward.simulate(flipped_ys)
            fltr_bkwd = np.flip(fltr_bkwd, axis=0)
            recv[trajNum, :, :] = (fltr_fwd + fltr_bkwd) / 2
        return recv

class LSOptModel:
    def __init__(self, A, C, Q, R):
        self.name = "LeastSquaresOpt"
        self.A, self.C, self.Q, self.R = A, C, Q, R
        
    def __call__(self, ys):
        num_traj, traj_len, obs_dim = ys.shape
        state_dim = self.A.shape[0]
        recv = np.zeros(shape=(num_traj, state_dim, traj_len))
        for traj_num in range(num_traj):
            recv[traj_num, :, :] = optimal_traj(self.A, self.C, self.Q, self.R, ys[traj_num, :, :].T)
        return recv

class GPTModel(nn.Module):
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
        embeds = self._read_in(ys)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction
    
class BERTModel(nn.Module):
    def __init__(self, n_dims_obs, n_dims_state, n_positions, n_embd, n_layer, n_head):
        super(BERTModel, self).__init__()
        self.name = "BERTModel"
        self.n_dims_obs = n_dims_obs
        self.n_dims_state = n_dims_state
        self._read_in = nn.Linear(n_dims_obs, n_embd).to(myDevice, dtype=torch.float32)
        self._backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=n_embd, dropout=0, activation="relu"), 
            num_layers=n_layer, norm=nn.LayerNorm(n_embd)
        )
        self._read_out = nn.Linear(n_embd, n_dims_state).to(myDevice, dtype=torch.float32)

    def forward(self, ys):
        import ipdb; ipdb.set_trace()
        embeds = self._read_in(ys)
        output = self._backbone(embeds)
        prediction = self._read_out(output)
        return prediction