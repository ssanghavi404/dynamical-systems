import copy
import numpy as np
import torch
from helpers import system_id
from plotting import *

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KFilter:
    def __init__(self, A, C, Q, R, state=None):
        self.A = A
        self.C = C
        self.Q = Q # covariance of v (process noise) 
        self.R = R # covariance of w (sensor noise)

        self.state_size = A.shape[0] 
        self.obs_size = C.shape[0]
        
        if state is None: self.state = np.zeros(self.state_size)
        else: self.state = state

        self.prev_P = np.zeros((self.state_size, self.state_size))
        self.P = np.zeros((self.state_size, self.state_size)) # covariance of observation at time t+1 given time t
        self.steady_state = False

        self.K = None # Kalman Gain, recalculated in update() function
    
    def measure(self):
        return self.C @ self.state

    def predict(self):
        self.prev_P = copy.deepcopy(self.P)
        self.state = self.A @ self.state
        self.P = self.A @ self.prev_P @ self.A.T + self.Q
        
    def update(self, measurement, t=0):
        if not self.steady_state:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
            self.P = (np.eye(self.state_size) - self.K @ self.C) @ self.P
            if np.allclose(self.P, self.prev_P): 
                self.steady_state = True
        innovation = measurement - self.C @ self.state
        self.state = self.state + self.K @ innovation

    def simulate(self, measurements):
        T = measurements.shape[0]
        states = np.zeros(shape=(T, self.state_size))
        for t in range(T):
            self.predict()
            self.update(measurements[t], t)
            states[t] = self.state
        return states
    
    def run_till_ss(self):
        state_init = copy.deepcopy(self.state)
        i = 0
        while not self.steady_state:
            i += 1
            self.predict()
            self.update(np.zeros(self.obs_size,))
        self.state = state_init
        return i


# class LearnedKF:
#     def __init__(self, state_dim, obs_dim=None, x0=None, optim='adam', lr=1e-2):
#         if obs_dim is None: obs_dim = state_dim
#         self.state_size = state_dim
#         self.obs_size = obs_dim

#         self.A = torch.eye(self.state_size, self.state_size, device=myDevice, requires_grad=True)
#         self.K = torch.zeros(self.obs_size, self.state_size, device=myDevice, requires_grad=True)
#         self.C = torch.eye(self.state_size, self.state_size, device=myDevice, requires_grad=True)

#         self.starting_state = torch.from_numpy(x0) if x0 is not None else torch.zeros(self.state_size, requires_grad=True)

#         self.loss_func = torch.nn.MSELoss()
#         if optim == 'adam': self.optimizer = torch.optim.Adam([self.A, self.C, self.K], lr=lr) 
#         else: self.optimizer = torch.optim.SGD([self.A, self.C, self.K], lr=lr)
#         self.losses = []

#     def measure(self, curr_state):
#         return self.C @ curr_state

#     def predict(self, curr_state):
#         nextState = self.A @ curr_state 
#         self.P_hat = self.A @ self.P @ self.A.T + self.Q
#         return nextState
    
#     def update(self, curr_state, obs):
#         self.K = self.P_hat @ self.C @ (self.C @ self.P_hat @ self.C.T + self.R)
#         self.P = (torch.eye(self.obs_size) - self.K @ self.C) @ self.P
#         next_state = curr_state + self.K @ (obs - self.measure(curr_state))
#         return next_state

#     def fit(self, meas, maxIt=15000, eps=1e-6, delta=1e-8):
#         '''Learn the Kalman Filter parameters from a single sequence of measurements and inputs'''
#         T, _ = meas.shape
#         meas_torch = torch.tensor(meas, requires_grad=False, device=myDevice)

#         prev_avg_seq_loss = 0 
#         avg_seq_loss = float('inf')

#         stopping_condition = False
#         i = 0

#         while not stopping_condition:
#             prev_avg_seq_loss = avg_seq_loss
#             seq_loss = None
#             curr_estimate = self.starting_state

#             for t in range(T-1):
#                 next_state = self.predict(curr_estimate)
#                 target = meas_torch[t]
#                 curr_loss = self.loss_func(self.measure(next_state), target)
#                 if seq_loss is None: seq_loss = curr_loss
#                 else: seq_loss += curr_loss
#                 curr_estimate = self.update(next_state, meas_torch[t])

#             self.optimizer.zero_grad()
#             seq_loss.backward()
#             self.optimizer.step()

#             i += 1
#             avg_seq_loss = seq_loss.item() / T
#             self.losses.append(avg_seq_loss)

#             if avg_seq_loss < eps: 
#                 print("Stopping because avg_seq_loss < eps"); stopping_condition = True
#             elif i > maxIt:
#                 print("Stopping due to maximum iterations hit"); stopping_condition = True
#             elif abs(avg_seq_loss - prev_avg_seq_loss) < delta: 
#                 print("Stopping because loss is not decreasing"); stopping_condition = True
#         print("LearnedKF converged in %d iterations" % i)

#     def simulate(self, measurements):
#         # 'measurements' is an T x obs_size array, where T is the number of timesteps
#         T = measurements.shape[0]
#         states = np.zeros(shape=(T, self.state_size))
#         curr_state = torch.zeros(self.state_size)
#         measurements = torch.tensor(measurements)
#         if inputs is not None: inputs = torch.tensor(inputs)
#         for t in range(T):
#             curr_state = self.predict(curr_state)
#             curr_state = self.update(curr_state, measurements[t])
#             states[t] = curr_state.detach().cpu().numpy()
#         return states

