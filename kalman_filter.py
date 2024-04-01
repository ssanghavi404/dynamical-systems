# Adapted from UC Berkeley EE 126
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

        self.state_size, self.obs_size = A.shape[0], C.shape[0] 
        self.state = state if state is not None else np.zeros(self.state_size)

        self.prev_P = np.eye(self.state_size, self.state_size)        
        self.P = np.eye(self.state_size, self.state_size) # covariance of observation at time t+1 given time t

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
        T = measurements.shape[1]
        states = np.zeros(shape=(self.state_size, T))
        for t in range(T):
            self.predict()
            states[:, t] = self.state
            self.update(measurements[:, t], t)
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