def visualize(trained_model, args, A=None, C=None, Q=None, R=None, x0=None, state_dim=None, obs_dim=None):
    trained_model.eval()


    for traj_len in [100, 200, 500] #, 1000, 2000, 5000, 10000]
    state_path = os.path.join(args.training.save_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_trajlen{6}_singleSys.pt'.format(args.model.family, args.task.order_n or args.task.matrix_dim, args.model.n_embd, args.model.n_head, args.model.n_layer, args.training.lr, traj_len))

    if os.path.exists(state_path): 
        state = torch.load(state_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        starting_step = state['train_step']

    if A is not None:
        x, y = generate_traj(args.num_tests, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)
    else:
        x, y = np.zeros(shape=(args.num_tests, args.task.order_n, args.task.traj_len)), np.zeros(shape=(args.num_tests, args.task.order_n, args.task.traj_len))

        # Different systems for each test trajectory
        for traj_num in range(args.num_tests):
            if args.task.env_name == 'stable_sys': A, C, Q, R, x0, state_dim, obs_dim = stable_sys_params(rng, order_n=args.task.order_n)
            elif args.task.env_name == 'nontrivial_sys': A, C, Q, R, x0, state_dim, obs_dim = nontrivial_sys_params(rng, order_n=args.task.order_n)
            elif args.task.env_name == 'so2': A, C, Q, R, x0, state_dim, obs_dim = so2_params()
            x[traj_num], y[traj_num] = generate_traj(1, args.task.traj_len, A, C, Q, R, x0, rng, state_dim, obs_dim)   

    input_ys = torch.tensor(y[:, :, :-1]).to(myDevice, dtype=torch.float32).permute(0, 2, 1)
    recv = trained_model(input_ys).detach().cpu().numpy()[:, 1:, :]

    if args.model.family == 'gpt': true_values = torch.tensor(y[:, :, 1:-1]).permute(0, 2, 1).detach().cpu().numpy()
    elif args.model.family == 'bert': true_values = torch.tensor(x[:, :, :-1]).permute(0, 2, 1).detach().cpu().numpy()

    errs = np.linalg.norm(recv - true_values, axis=2)

    # Plot the errors over time
    metrics = {"Transformer": {'med': np.quantile(errs, 0.5, axis=0),
                              'q1': np.quantile(errs, 0.25, axis=0),
                              'q3': np.quantile(errs, 0.75, axis=0)}}
    taskname = {'gpt':'nextstate_prediction', 'bert':'truestate_recovery'}[args.model.family]

    plt.figure()
    plt.plot(*true_values[0].T, label="Ys")
    # plt.plot(*torch.tensor(x).permute(0, 2, 1).detach().cpu().numpy()[0].T[:2], label="Xs")
    plt.plot(*input_ys[0].T, label='Input Ys')
    plt.plot(*recv[0].T, label="Recovered by Transformer")
    for baseline_model in get_baselines(taskname, A, C, Q, R, x[:, :, 0], x[:, :, -1]):
        recv = baseline_model(input_ys.detach().cpu().numpy())
        if args.model.family == 'gpt':
            errs = np.linalg.norm(recv[:, :-1, :] - true_values, axis=2)
        elif args.model.family == 'bert':
            errs = np.linalg.norm(np.array(recv - true_values, axis=2))
        
        print("errs has shape", errs.shape)
        plt.plot(*recv[0].T, label="Recovered by {0}".format(baseline_model.name))
        
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