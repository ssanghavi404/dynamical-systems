import numpy as np
import cvxpy as cp

# Helper function to perform system identification. Works for systems where the C is assumed to be the identity (outputs are noisy direct measurements of the input states).
def system_id(meas, t, x0=0):
    '''system_id(measurement_data, t, starting_state, inputs_data) -> A, B
    Performs system identification using the first t timesteps of data.
    '''
    state_dim = meas[0].shape[0]
    # input_dim = 0 if inputs is None else inputs.shape[1]

    M = np.zeros(shape=(t*state_dim, state_dim*(state_dim)), dtype=np.float64) # shape=(t*state_dim, state_dim*(state_dim+input_dim))
    c = np.zeros(shape=(t*state_dim, 1))
    
    for i in range(t):
        for n in range(state_dim):
            row = i*state_dim + n
            c[row] = meas[i, n] 
            M[row, n*state_dim:(n+1)*state_dim] = meas[i-1] if i > 0 else x0
    A_found = np.linalg.lstsq(M, c, rcond=None)[0].T[0].reshape((state_dim, state_dim))
    return A_found

# Helper function to calculate least-squares optimized trajectory, minimum energy noise
def optimal_traj(A, C, Q, R, y):
    Qinv = np.linalg.inv(Q)
    Rinv = np.linalg.inv(R)
    state_dim = A.shape[0]
    T = y.shape[0]
    xs = cp.Variable((T, state_dim))
    # Set up the objective function
    obj = 0
    for i in range(1, T):
        w_hyp = xs[i, :] - A @ xs[i-1, :]
        obj += cp.quad_form(w_hyp, Qinv) # Minimize sum of process noises...
    for i in range(T):
        v_hyp = y[i, :] - C @ xs[i, :]
        obj += cp.quad_form(v_hyp, Rinv) # ...and sum of sensor noises
    # Special handling for the first state
    w_hyp0 = xs[0, :] - A @ y[0, :]
    obj += cp.quad_form(w_hyp0, Qinv)

    # Solve CVXPY problem with the objective above.
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    ls_rec = xs.value

    return ls_rec
