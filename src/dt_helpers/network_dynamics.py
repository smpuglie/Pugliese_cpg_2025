import numpy as np
from typing import Callable
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal.windows import blackman


def run_dynamics_step(
    current_rates: np.ndarray,
    network_input: np.ndarray,
    weights: np.ndarray,
    taus: np.ndarray | np.float64 | np.float32 | float,
    dt: np.float64 | np.float32 | float,
    noise: np.ndarray | np.float64 | np.float32 | float = 0,
    activations: Callable | list[Callable] = lambda x: x,
) -> np.ndarray:
    """tau * dr / dt = -r + phi(W * r + I) => dr = dt * (-r + phi(W * r + I)) / tau

    Args:
        current_rates (np.ndarray): Current rates of the network
        network_input (np.ndarray): Input to the network
        weights (np.ndarray): Weights of the network
        taus (np.ndarray | np.float64 | np.float32 | float): Time constants of the network
        dt (np.float64 | np.float32 | float): Time step
        noise (np.ndarray | np.float64 | np.float32 | float, optional): Noise to add to the network. Defaults to 0
        activations (_type_, optional): Activation functions of the network. Defaults to <lambda x: x>

    Returns:
        np.ndarray: Updated rates of the network
    """

    time_factor = dt / taus
    activations = (
        activations
        if isinstance(activations, Callable)
        else lambda x: np.array([activations[i](x[i]) for i in range(len(x))])
    )
    return (1 - time_factor) * current_rates + time_factor * activations(
        weights @ current_rates + network_input + noise
    )


def compute_individual_neuron_frequency(
    network_activity: np.ndarray,
    dt: float,
    num_steps: int,
    neuron_names: list[str] = None,
    max_frequency: float = 5,
    use_interpolation: bool = False,
) -> dict:
    neuron_names = neuron_names if neuron_names is not None else [str(i) for i in range(network_activity.shape[1])]
    individual_neuron_frequencies = dict()
    xs_f = fftfreq(num_steps, dt)[: num_steps // 2]
    if use_interpolation:
        xs_f_interp = np.linspace(0, xs_f[-1], len(xs_f) * 20)

    weights = blackman(num_steps)
    for i in range(network_activity.shape[1]):
        ys = network_activity[:, i]
        ys_fft = fft(ys * weights)
        ys_fft = np.abs(ys_fft[0 : num_steps // 2])
        peaks, _ = find_peaks(ys_fft)

        individual_neuron_frequencies[neuron_names[i]] = 0
        if len(peaks) >= 1:
            temp_freq = xs_f[peaks[ys_fft[peaks].argmax()]]

            if use_interpolation:
                interp_fun = interp1d(xs_f, ys_fft, kind="cubic")
                ys_fft_interp = interp_fun(xs_f_interp)
                peaks_interp, _ = find_peaks(ys_fft_interp)
                temp_freq = xs_f_interp[peaks_interp[np.abs(xs_f_interp[peaks_interp] - temp_freq).argmin()]]

            if temp_freq < max_frequency:
                individual_neuron_frequencies[neuron_names[i]] = temp_freq

    return individual_neuron_frequencies
