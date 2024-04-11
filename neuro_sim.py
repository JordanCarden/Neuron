import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate LIF neuron model and alpha synapse model.")
    parser.add_argument("mode", choices=["spike", "current"], help="Simulation mode: 'spike' or 'current'")
    parser.add_argument("sim_time", type=float, help="Simulation time in milliseconds")
    parser.add_argument("--spike_rate", type=int, help="Input spike rate in Hz (only for 'spike' mode)")
    parser.add_argument("--current", type=float, help="Input current in nanoamps (only for 'current' mode)")
    return parser.parse_args()

def load_config():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    return config

def lif_neuron_sim(config, sim_time, input_current):
    """
    Simulate a Leaky Integrate-and-Fire (LIF) neuron.

    Args:
        config (dict): Configuration parameters.
        sim_time (float): Simulation time in milliseconds.
        input_current (float): Input current in nanoamps.

    Returns:
        list: List of membrane voltages over time.
    """
    # Extract configuration parameters
    rest_potential = config["v_r"]
    spike_threshold = config["v_thr"]
    spike = config["v_spike"]
    refractory_period = config["t_r"]
    delta_t = config["dt"]

    # Initialize variables
    last_spike_time = 0
    membrane_voltage = [spike]

    # Perform simulation using Euler's method
    for time in np.arange(0, sim_time/1000, delta_t):
        # Check if the neuron is in the refractory period
        if time - last_spike_time <= refractory_period:
            voltage = rest_potential
        else:
            voltage = spike

        membrane_voltage.append(voltage)

        if voltage >= spike_threshold:
            last_spike_time = time
            voltage = rest_potential

    return membrane_voltage


def alpha_synapse_sim(config, sim_time, spike_times):
    """
    Simulate an alpha synapse.

    Args:
        config (dict): Configuration parameters.
        sim_time (float): Simulation time in milliseconds.
        spike_times (list): List of spike times.

    Returns:
        ndarray: Array of synaptic currents over time.
    """
    # Extract configuration parameters
    reversal_potential = config["reversal_potential"]
    alpha_synapse_weight = config["alpha_synapse_weight"]
    alpha_synapse_max_conductance = config["alpha_synapse_max_conductance"]
    alpha_synapse_decay_time_constant = config["alpha_synapse_decay_time_constant"]
    delta_t = config["delta_t"]

    # Initialize synaptic current array
    synaptic_current = np.zeros(int(sim_time / delta_t))

    # Perform simulation
    for spike_time in spike_times:
        for i, time in enumerate(np.arange(0, sim_time, delta_t)):
            if time >= spike_time:
                # Calculate synaptic current using alpha synapse model
                synaptic_current[i] += alpha_synapse_weight * \
                                       alpha_synapse_max_conductance * \
                                       (reversal_potential - synaptic_current[i]) * \
                                       (time - spike_time) * \
                                       np.exp(- (time - spike_time) / alpha_synapse_decay_time_constant)

    return synaptic_current

def main():
    args = parse_arguments()
    config = load_config()

    if args.mode == 'spike':
        if args.spike_rate is None:
            parser.error('--spike_rate is required for "spike" mode')
        spike_rate = args.spike_rate
        spike_times = np.arange(0, args.sim_time / 1000, 1 / spike_rate)
        synaptic_current = alpha_synapse_sim(config, args.sim_time, spike_times)
    else:  # args.mode == 'current'
        if args.current is None:
            parser.error('--current is required for "current" mode')
        input_current = args.current
        membrane_voltage = lif_neuron_sim(config, args.sim_time, input_current)

    if args.mode == 'spike':
        plt.plot(spike_times, synaptic_current)
        plt.xlabel('Time (s)')
        plt.ylabel('Synaptic Current (A)')
        plt.title('Synaptic Current Over Time')
        plt.grid(True)
        plt.show()
    else:  # args.mode == 'current'
        plt.plot(membrane_voltage)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        plt.title('Membrane Voltage Over Time')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
