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

def lifFunction(previousVoltage, inputCurrent):
    config = load_config()
    rest_potential = config["v_r"]
    delta_t = config["dt"]
    decay_time = config["tao_m"]
    membrane_capacity = config["c_m"]

    return previousVoltage + delta_t * (((-previousVoltage + rest_potential) / decay_time) + (inputCurrent/(membrane_capacity*1000000000)))

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
    decay_time = config["tao_m"]
    membrane_capacity = config["c_m"]

    # Initialize variables
    last_spike_time = 0
    membrane_voltage = [spike]

    # Perform simulation using Euler's method
    for time in np.arange(0, sim_time/1000, delta_t):
        # Check if the neuron is in the refractory period
        if time - last_spike_time <= refractory_period:
            voltage = rest_potential
        else:
            voltage = lifFunction(membrane_voltage[-1], input_current)
            print(voltage)
            if voltage >= spike_threshold:
                last_spike_time = time
                voltage = spike

        membrane_voltage.append(voltage)

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
    reversal_potential = config["v_rev"]
    alpha_synapse_weight = config["w"]
    alpha_synapse_max_conductance = config["alpha_synapse_max_conductance"]
    alpha_synapse_decay_time_constant = config["tao_syn"]
    delta_t = config["dt"]

    # Initialize synaptic current array
    membrane_voltage = [0]

    for time in np.arange(0, sim_time/1000, delta_t):
        # Check if the neuron is in the refractory period
        if time - last_spike_time <= refractory_period:
            voltage = rest_potential
        else:
            voltage = lifFunction(membrane_voltage[-1], input_current)
            print(voltage)
            if voltage >= spike_threshold:
                last_spike_time = time
                voltage = spike

        membrane_voltage.append(voltage)

    return membrane_voltage

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
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Membrane Voltage Over Time')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
