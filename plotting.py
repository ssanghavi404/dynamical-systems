import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

def plot(metrics, title, num_iterations):
    fig, ax = plt.subplots(1, 1)

    color = 0
    for name, values in metrics.items():
        ax.plot(values["med"], "-", label=name, color=palette[color % 10], lw=2)
        low = values["q1"]
        high = values["q3"]
        ax.fill_between(range(len(low)), low, high, alpha=0.2)
        color += 1
    ax.set_xlabel("Timestep in Trajectory")
    ax.set_ylabel("Median Error at Timestep (IQR shaded)")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_title(label=title)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    plt.savefig('gpt_iterations{0}'.format(num_iterations))
    plt.show()
    return fig, ax