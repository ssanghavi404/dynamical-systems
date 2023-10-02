import matplotlib.pyplot as plt

# Generic wrapper so that all plotting in the original notebook can use this
def plot(series_dict: dict):
    T = None
    num_dims = float('inf') 
    for title, traj in series_dict.items():
        print(title, "shape", traj.shape)
        T = traj.shape[0]
        num_dims = min(num_dims, traj.shape[1])
    if num_dims == 1: plot_vs_time(T, series_dict)
    elif num_dims == 2: plot_poses2d(series_dict)
    elif num_dims == 3: plot_poses3d(series_dict)
    else: raise AssertionError("Number of dimensions to plot should be at most 3")

# Plots 2d poses in Cartesian space
def plot_poses2d(series_dict: dict):
    plt.figure()
    ax = plt.axes()
    for title, traj in series_dict.items():
        ax.plot(traj[:, 0], traj[:, 1], label=title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.legend()
    plt.show()

# Plots 3d poses in Cartesian space
def plot_poses3d(series_dict: dict):
    plt.figure()
    ax = plt.axes(projection ='3d') 
    for title, traj in series_dict.items():  
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], label=title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend()
    plt.show()

# Plots the "ind"-th element of the state variable vs. time. The default for ind is 0.
def plot_vs_time(T: int, series_dict: dict, ind=0):
    plt.figure()
    ax = plt.axes()
    for title, traj in series_dict.items():
        ax.plot(range(T), traj[:, ind], label=title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("x")
    plt.legend()
    plt.show()