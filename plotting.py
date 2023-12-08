import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

def plot(metrics, args, title):
    fig, ax = plt.subplots(1, 1)

    color = 0
    for name, values in metrics.items():
        ax.plot(values["med"], "-", label=name, color=palette[color % 10], lw=2)
        low = values["q1"]
        high = values["q3"]
        ax.fill_between(range(len(low)), low, high, alpha=0.2)
        color += 1
    ax.set_xlabel("Timestep in Trajectory")
    ax.set_ylabel("Median Error at Timestep")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_title(label=title)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    graph_path = os.path.join(args.training.graph_path, '{0}_order{1}_emb{2}_heads{3}_layers{4}_lr{5}_stableSys{6}_numIt{7}.jpg'.format(
                                args.model.family, args.training.order_n, args.model.n_embd, args.model.n_head, 
                                args.model.n_layer, args.training.lr, args.training.env_name, args.training.num_iterations))
    plt.savefig(graph_path)
    plt.show()
    return fig, ax