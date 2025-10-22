from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_spikes(
    spike_activity: np.ndarray,
    membrane_potential: np.ndarray,
    duration: float,
    dt: float,
    num_steps: Optional[int] = None,
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
    fig, axes = plt.subplots(
        nrows, 1, figsize=(20, ax_height_ratios.sum()), sharex=True, gridspec_kw={"height_ratios": ax_height_ratios}
    )
    if nrows == 1:
        axes = [axes]
    time_indices = (np.arange(num_steps) + 1) * dt
    for i in range(spike_activity.shape[1]):
        axes[0].scatter(  # type: ignore
            time_indices[spike_activity[:, i]],
            spike_activity[spike_activity[:, i], i] * (i + 1),
            marker="|",
            s=marker_size,
            label=neuron_labels[i],
        )
    axes[0].set_ylabel("Spikes", fontsize=12)  # type: ignore
    axes[0].set_yticks(np.arange(1, spike_activity.shape[1] + 1), neuron_labels)  # type: ignore
    axes[0].grid()  # type: ignore
    sns.despine(ax=axes[0], offset={"bottom": 0, "left": 10}, trim=True)  # type: ignore

    xs = (np.arange(num_steps) + 1) * dt
    for i in range(nrows - 1):
        axes[i + 1].plot(xs, membrane_potential[:, i], c=f"C{i}")  # type: ignore
        axes[i + 1].plot([xs[0], xs[-1]], [v_rest, v_rest], "k--", lw=1.5)  # type: ignore
        axes[i + 1].plot([xs[0], xs[-1]], [v_threshold, v_threshold], "k--", lw=1.5)  # type: ignore
        axes[i + 1].set_ylabel(  # type: ignore
            f"{neuron_labels[i]}\n" r"$V_m$ [mV]", fontsize=12, rotation=y_label_rotation, ha="right", va="center"
        )
        axes[i + 1].grid()  # type: ignore

    axes[-1].set_xlabel("Time (s)", fontsize=12)  # type: ignore
    axes[-1].set_xlim(0, duration)  # type: ignore
    if show_plot:
        plt.show()
    else:
        return fig, axes
