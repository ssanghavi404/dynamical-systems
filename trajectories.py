import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import unittest

from plotting import *

# Given the system parameters and timelength, generate num_traj
def generate_traj(num_traj, T, A, C, Q, R, x0, rng, state_dim=None, obs_dim=None):
    if state_dim is None: state_dim = A.shape[0]
    if obs_dim is None: obs_dim = C.shape[0]

    x = np.zeros(shape=(num_traj, state_dim, T+1))
    for i in range(num_traj): x[i, :, 0] = x0

    y = np.zeros(shape=(num_traj, obs_dim, T+1))

    for i in range(num_traj):
        for t in range(T):
            w = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q)
            v = rng.multivariate_normal(mean=np.zeros(obs_dim), cov=R)
            x[i, :, t+1] = A @ x[i, :, t] + w
            y[i, :, t] = C @ x[i, :, t] + v
        v = rng.multivariate_normal(mean=np.zeros(obs_dim), cov=R)
        y[i, :, T] = C @ x[i, :, T] + v

    return x, y

# 2d rotation around a circle, rotate "angle" degrees in each timestep
def so2_params(angle=1, process_noise=0.001, sensor_noise=0.01):
    # Rotate "angle" degrees in each timestep
    theta = angle * 1/360*2*np.pi # one degree per timestep
    state_dim = 2
    obs_dim = 2
    A = np.array([[np.cos(theta), -np.sin(theta)], # state transition matrix
                  [np.sin(theta),  np.cos(theta)]]) # moving around a circle at 1 deg per timestep
    C = np.eye(obs_dim, state_dim) # State fully observed
    Q = process_noise * np.eye(state_dim) # Covariance matrix of process noise
    R = sensor_noise * np.eye(obs_dim) # Covariance matrix of sensor noise
    x0= np.array([1.0, 0.0], dtype=np.float64) # Starting state
    return A, C, Q, R, x0, state_dim, obs_dim

def so3_params(angle=1, axis=np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]), process_noise=0.001, sensor_noise=0.01):
    theta = angle * 1/360*2*np.pi # one degree per timestep
    state_dim = 3
    obs_dim = 3

    # Default rotation axis is middle of first quadrant
    A = Rotation.from_rotvec(axis * theta).as_matrix() 
    C = np.eye(state_dim)
    Q = process_noise * np.eye(state_dim)
    R = sensor_noise * np.eye(obs_dim)
    x0= np.array([1.0, 0.0, 0.0], dtype=np.float64) # starting state
    return A, C, Q, R, x0, state_dim, obs_dim

def smd_params(mass=1, k_spring=1, b_damper=0.2, process_noise=0.0001, sensor_noise=0.01):
    state_dim = 2
    obs_dim = 1
    m = mass # Mass
    k = k_spring # Spring Constant
    b = b_damper # Damping

    # State space is [[x], [xdot]]
    Ac = np.array([[ 0.0, 1.0], 
                   [-k/m, -b/m]]) # Continuous time dynamics
    Bc = np.array([[0.0], [1/m]]) # Continuous time input transformation

    # model discretization
    sampling = 0.05 # sampling interval
    A = np.linalg.inv(np.eye(state_dim) - sampling*Ac)

    C = np.array([[1.0, 0.0]]) # Only position x observed at each timestep
    Q = process_noise * np.eye(state_dim)
    R = sensor_noise * np.eye(obs_dim)
    x0= np.array([1.0, 0.0]) # Starting state
    return A, C, Q, R, x0, state_dim, obs_dim

# Traveling with a constant velocity that can be driven, only the position is observed.
def motion_params(process_noise=1, sensor_noise=0.4):
    state_dim = 2
    obs_dim = 1
    dt = 1e-3

    # State space is [[x], [xdot]]
    A = np.array([[1, dt], 
                  [0, 1]])
    C = np.array([[1, 0]]) # only observe the position x
    Q = process_noise * np.array([[0, 0], 
                                  [0, 1]])
    R = sensor_noise * np.array([[1]])
    x0= np.array([0.0, 0.0], dtype=np.float64) 
    return A, C, Q, R, x0, state_dim, obs_dim

# Falling with constant acceleration. Velocity can be driven. Only position is observed.
# Process noise only affects the velocity. Sensor noise on the position.
def accel_params(start_height=20, accel=-10, process_noise=0.001, sensor_noise=0.1):
    state_dim = 3
    obs_dim = 1
    dt = 1e-3
    
    # State space is [[x], [xdot], [xdotdot]]
    A = np.array([[1, dt, 0], 
                  [0, 1, dt], 
                  [0, 0, 1]])
    C = np.array([[1, 0, 0]]) # only the position can be observed
    Q = process_noise * np.eye(state_dim)
    R = sensor_noise * np.eye(obs_dim)
    x0= np.array([start_height, 0, accel]) 
    return A, C, Q, R, x0, state_dim, obs_dim

# i-th order stable system
def stable_sys_params(rng, order_n=3, process_noise=1e-3, sensor_noise=1e-4):
    # Generate a stable or marginally stable system of state_dim, with input_dim-dimensional inputs
    diag = np.zeros(shape=(order_n, order_n))

    # # All complex eigenvalues should come in complex conjugate pairs.
    for i in range(order_n // 2):
        r = rng.random()*2 - 1 # Between -1 and 1
        theta = rng.random() * np.pi # random angle between 0 and 180deg
        block = r * np.array([[ np.cos(theta), -np.sin(theta) ],  # rotation matrix
                              [ np.sin(theta),  np.cos(theta) ]]) 
        diag[2*i:2*(i+1),  2*i:2*(i+1)] = block
    
    # if n is odd: need to have one real eigenvalue
    if order_n % 2 == 1: diag[-1, -1] = rng.random()*2 - 1
    
    P = rng.normal(size=(order_n, order_n))

    state_dim = order_n
    obs_dim = order_n
    A = P @ diag @ np.linalg.inv(P)
    C = np.eye(obs_dim, state_dim) # State fully observed
    Q = process_noise * np.eye(state_dim) # Covariance matrix of process noise
    R = sensor_noise * np.eye(obs_dim) # Covariance matrix of sensor noise
    rand = rng.random()
    if rand < 0.2: x0 = np.zeros(order_n, dtype=np.float64); # Starting state at zero
    elif rand < 0.7: # starting state at steady state (start at 0, burn 100 samples)
        x0 = np.zeros(order_n, dtype=np.float64) # Starting state
        for i in range(100):
            w_t = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q) # process noise
            x0 = A @ x0 + w_t
    else: x0 = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q)
    x0 = np.zeros(state_dim)
    return A, C, Q, R, x0, state_dim, obs_dim

# order_n system with nontrivial jordan blocks
def nontrivial_sys_params(rng, order_n=3, process_noise=1e-3, sensor_noise=1e-4):
    if rng.random() < 0.2: P, S, Vt = np.linalg.svd(rng.random(size=(order_n, order_n))*2 - 1); # random unitary matrix
    else: P = rng.normal(size=(order_n, order_n))
    diag = np.zeros(shape=(order_n, order_n))
    curr_eig_index = 0
    while curr_eig_index < order_n: 
        if curr_eig_index == order_n - 1: # If we only have one more eigenvalue to generate, it must be real.
            eig = (rng.random() * 2 - 1)
            diag[curr_eig_index, curr_eig_index] = eig # random number between -1 and 1
            curr_eig_index += 1
        else:
            if rng.random() < 0.3: # With probability 30%, draw a real eigenvalue.
                eig = rng.random() * 2 - 1 # random number between -1 and 1
                mult = order_n+1
                while mult + curr_eig_index > order_n: # rejection sample if we go over
                    mult = rng.poisson(lam=1) + 1 # Poisson random variable to determine the multiplicity of the eigenvalue
                # print("Placing eigenvalue", eig, "with multiplicity", mult)
                for i in range(mult):
                    diag[curr_eig_index, curr_eig_index] = eig
                    if i > 0: diag[curr_eig_index-1, curr_eig_index] = 1 # add a "1" above the diagonal for nontrivial entries
                    curr_eig_index += 1
            else:
                r = rng.random()*2 - 1 # Between -1 and 1
                theta = rng.random() * np.pi # random angle between 0 and 180deg
                # print('Placing eigenvalues', r * np.cos(theta) , " +/- " , r * np.sin(theta), "j")
                # Complex eigenvalues in a conjugate pair
                block = r * np.array([  [ np.cos(theta), -np.sin(theta) ],  # rotation matrix
                                        [ np.sin(theta),  np.cos(theta) ]   ]) 
                diag[curr_eig_index:(curr_eig_index+2),  curr_eig_index:(curr_eig_index+2)] = block
                curr_eig_index += 2
    A = P @ diag @ np.linalg.inv(P)
    state_dim = order_n
    obs_dim = order_n
    C = np.eye(obs_dim, state_dim) # State fully observed
    Q = process_noise * np.eye(state_dim) # Covariance matrix of process noise
    R = sensor_noise * np.eye(obs_dim) # Covariance matrix of sensor noise
    x0 = np.zeros(order_n, dtype=np.float64) # Starting state
    return A, C, Q, R, x0, state_dim, obs_dim

class TestSystems(unittest.TestCase):

    def test_so2_params(self):
        A, C, Q, R, x0, state_dim, obs_dim = so2_params(angle=1, process_noise=0.001, sensor_noise=0.01)
        self.assertTrue(np.allclose(A, np.array([[0.999847695, -0.017452406], [0.017452406, 0.999847695]])))
        self.assertTrue(np.allclose(C, np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.allclose(Q, np.array([[0.001, 0], [0, 0.001]])))
        self.assertTrue(np.allclose(R, np.array([[0.01, 0], [0, 0.01]])))
        self.assertTrue(np.allclose(x0, np.array([1.0, 0.0])))
        self.assertEqual(state_dim, 2)
        self.assertEqual(obs_dim, 2)

    def test_so3_params(self):
        A, C, Q, R, x0, state_dim, obs_dim = so3_params(angle=1, axis=np.array([1, 0, 0]), process_noise=0.001, sensor_noise=0.01)
        self.assertTrue(np.allclose(A, np.array([[1, 0, 0], [0, 0.999847695, -0.017452406], [0, 0.017452406, 0.999847695]])))
        self.assertTrue(np.allclose(C, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        self.assertTrue(np.allclose(Q, np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])))
        self.assertTrue(np.allclose(R, np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])))
        self.assertTrue(np.allclose(x0, np.array([1., 0., 0.])))
        self.assertEqual(state_dim, 3)
        self.assertEqual(obs_dim, 3)

    def test_smd_params(self):
        A, C, Q, R, x0, state_dim, obs_dim = smd_params()
        self.assertTrue(np.allclose(A, np.array([[ 0.99753086,  0.04938272],
                                                 [-0.04938272,  0.98765432]])))
        self.assertTrue(np.allclose(C, np.array([[1, 0]])))
        self.assertTrue(np.allclose(Q, np.array([[0.0001, 0], [0, 0.0001]])))
        self.assertTrue(np.allclose(R, np.array([[0.01]])))
        self.assertTrue(np.allclose(x0, np.array([1.0, 0.0])))
        self.assertEqual(state_dim, 2)
        self.assertEqual(obs_dim, 1)

    def test_motion_params(self):
        A, C, Q, R, x0, state_dim, obs_dim = motion_params()
        self.assertTrue(np.allclose(A, np.array([[1.0, 0.001], [0.0, 1.0]])))
        self.assertTrue(np.allclose(C, np.array([[1, 0]])))
        self.assertTrue(np.allclose(Q, [[0, 0], [0, 1]]))
        self.assertTrue(np.allclose(R, [[0.4]]))
        self.assertTrue(np.allclose(x0, [0., 0.]))
        self.assertEqual(state_dim, 2)
        self.assertEqual(obs_dim, 1)

    def test_accel_params(self):
        A, C, Q, R, x0, state_dim, obs_dim = accel_params()
        self.assertTrue(np.allclose(A, np.array([[1, 0.001, 0], [0.0, 1.0, 0.001], [0.0, 0.0, 1.0]])))
        self.assertTrue(np.allclose(C, np.array([[1, 0, 0]])))
        self.assertTrue(np.allclose(Q, np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])))
        self.assertTrue(np.allclose(R, np.array([[0.1]])))
        self.assertTrue(np.allclose(x0, [20, 0, -10]))
        self.assertEqual(state_dim, 3)
        self.assertEqual(obs_dim, 1)

    def test_stable_sys_params(self):
        rng = np.random.default_rng(seed=0)
        A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng)
        self.assertTrue(np.allclose(A, np.array([[ 0.54467934, 0.80731238, 0.94897699],
                                                [ 0.0094852,  -0.99323337, -1.36323226],
                                                [-0.71595766, -0.32024279, -0.10692609]])))
        self.assertTrue(np.allclose(C, np.array([[1., 0., 0.], [0., 1., 0.], [0, 0, 1]])))
        self.assertTrue(np.allclose(Q, [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]]))
        self.assertTrue(np.allclose(R, [[0.0001, 0., 0.], [0., 0.0001, 0.], [0., 0., 0.0001]]))
        self.assertTrue(np.allclose(x0, [0., 0., 0.]))
        self.assertEqual(state_dim, 3)
        self.assertEqual(obs_dim, 3)

    def test_nontrivial_sys_params(self):
        rng = np.random.default_rng(seed=0)
        A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng)
        self.assertTrue(np.allclose(A, np.array([[0.34488042, -1.01989897, -0.94507362], 
                                                 [-0.94292546, -3.62689264, -2.85436305], 
                                                 [ 1.49062263, 4.80473937, 4.14196006]])))
        self.assertTrue(np.allclose(C, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        self.assertTrue(np.allclose(Q, np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]] )))
        self.assertTrue(np.allclose(R, np.array([[0.0001, 0, 0], [0, 0.0001, 0], [0, 0, 0.0001]] )))
        self.assertTrue(np.allclose(x0, np.array([0, 0, 0])))
        self.assertEqual(state_dim, 3)
        self.assertEqual(obs_dim, 3)

if __name__ == '__main__':
    unittest.main()