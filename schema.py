from quinine import tstring, tinteger, tfloat, allowed, required, nullable, default, stdict
from funcy import merge

model_schema = {
    'family': merge(tstring, allowed(['gpt', 'bert'])),
    'n_embd': merge(tinteger, required),
    'n_layer': merge(tinteger, required),
    'n_head': merge(tinteger, required),
}

task_schema = {
    # Environment, or Matrix to use. 
    # If env_name is specified, then task (single or general) must also be specified, and we will train a model to that (matrix args will be ignored)
    # If env_name is stable_sys or nontrivial_sys, then we must specify what order system we want to train
    # Otherwise, matrix_dim and matrix_type will load the specified matrix and train a model to that
    'env_name': merge(tstring, nullable, allowed([None, 'stable_sys', 'nontrivial_sys', 'so2']), default('so2')),
    'task': merge(tstring, allowed(['single', 'general']), default('single')),
    'order_n': merge(tinteger, nullable, default(3)),
    'matrix_dim': merge(tinteger, nullable, allowed([None, 2, 3, 4, 5, 6]), default(None)),
    'matrix_type': merge(tstring, nullable, allowed([None, 'diag', 'ident']), default(None)),

    # Trajectory Length
    'traj_len': merge(tinteger, default(200)),
}

training_schema = {
    # Training Hyperparameters
    'batch_size': merge(tinteger, default(64)),
    'lr': merge(tfloat, default(1e-3)),
    'optim': merge(tstring, allowed(['adam', 'adamw', 'sgd']), default('adamw')),
    'num_iterations': merge(tinteger, default(1000)),

    # Information about saving the models
    'save_every_steps': merge(tinteger, default(300)),  # how often to checkpoint
    'save_path': merge(tstring, default('./results/')), # where to save the results
    'graph_path': merge(tstring, default('./plots/')), # where to save plot figures. 
}

wandb_schema = {
    'project': merge(tstring, default('dynamical-systems')),
    'entity': merge(tstring, default('saagar')),
    'notes': merge(tstring, default('')),
    'log_every_steps': merge(tinteger, default(10)),
}

schema = {
    'model': stdict(model_schema),
    'task': stdict(task_schema),
    'training': stdict(training_schema),
    'wandb': stdict(wandb_schema),
    'num_tests': merge(tinteger, nullable, default(None)) # Number of tests to run in evaluation
}
