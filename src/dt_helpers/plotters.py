import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_network_activity(
    network_activity: np.ndarray,
    duration: np.float64 | np.float32 | float,
    dt: np.float64 | np.float32 | float,
    num_steps: int = None,
    title: str = "",
    y_label: str = "Rate (by neuron)",
    legend_labels: list[str] = ["Exc-input", "Exc-hidden", "Inh-hidden"],
    show_plot: bool = True,
):
    num_steps = int(duration / dt) if num_steps is None else num_steps
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot((np.arange(num_steps) + 1) * dt, network_activity)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlim([0, duration])
    ax.set_title(title, fontsize=15)
    ax.legend(legend_labels)
    plt.grid()
    sns.despine(ax=ax, offset={"bottom": 0, "left": 10}, trim=True)
    if show_plot:
        plt.show()
    else:
        return fig, ax


def visualize_spikes(
    spike_activity: np.ndarray,
    membrane_potential: np.ndarray,
    duration: np.float64 | np.float32 | float,
    dt: np.float64 | np.float32 | float,
    num_steps: int = None,
    neuron_labels: list[str] = ["a", "b", "c", "d"],
    show_plot: bool = True,
    **kwargs,
):
    marker_size = kwargs.get("spike_marker_size", 200)
    spike_figure_height = kwargs.get("spike_figure_height", 2)
    membrane_potential_figure_height = kwargs.get("membrane_potential_figure_height", 2)
    y_label_rotation = kwargs.get("y_label_rotation", 0)
    v_rest = kwargs.get("v_rest", -52)
    v_threshold = kwargs.get("v_threshold", -45)

    membrane_potential = membrane_potential * 1e3
    if len(neuron_labels) != spike_activity.shape[1]:
        neuron_labels = [str(i) for i in range(spike_activity.shape[1])]
        print("renamed neuron_labels to default")

    num_steps = int(duration / dt) if num_steps is None else num_steps
    nrows = 1 + membrane_potential.shape[1]
    ax_height_ratios = np.ones(nrows) * membrane_potential_figure_height
    ax_height_ratios[0] = spike_figure_height
    fig, ax = plt.subplots(
        nrows, 1, figsize=(20, ax_height_ratios.sum()), sharex=True, gridspec_kw={"height_ratios": ax_height_ratios}
    )
    time_indices = (np.arange(num_steps) + 1) * dt
    for i in range(spike_activity.shape[1]):
        ax[0].scatter(
            time_indices[spike_activity[:, i]],
            spike_activity[spike_activity[:, i], i] * (i + 1),
            marker="|",
            s=marker_size,
            label=neuron_labels[i],
        )
    ax[0].set_ylabel("Spikes", fontsize=12)
    ax[0].set_yticks(np.arange(1, spike_activity.shape[1] + 1), neuron_labels)
    ax[0].grid()
    sns.despine(ax=ax[0], offset={"bottom": 0, "left": 10}, trim=True)

    xs = (np.arange(num_steps) + 1) * dt
    for i in range(nrows - 1):
        ax[i + 1].plot(xs, membrane_potential[:, i], c=f"C{i}")
        ax[i + 1].plot([xs[0], xs[-1]], [v_rest, v_rest], "k--", lw=1.5)
        ax[i + 1].plot([xs[0], xs[-1]], [v_threshold, v_threshold], "k--", lw=1.5)
        ax[i + 1].set_ylabel(
            f"{neuron_labels[i]}\n" r"$V_m$ [mV]", fontsize=12, rotation=y_label_rotation, ha="right", va="center"
        )
        ax[i + 1].grid()

    ax[-1].set_xlabel("Time (s)", fontsize=12)
    ax[-1].set_xlim([0, duration])
    if show_plot:
        plt.show()
    else:
        return fig, ax
