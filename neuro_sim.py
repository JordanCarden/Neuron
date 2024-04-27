import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import math

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulate LIF neuron model and alpha synapse model.")
    parser.add_argument("mode", choices=["spike", "current"], help="Simulation mode: 'spike' or 'current'")
    parser.add_argument("simTime", type=float, help="Simulation time in milliseconds")
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

    return previousVoltage + delta_t * (((-previousVoltage + rest_potential) / decay_time) + (inputCurrent/(membrane_capacity)))

def lif_neuron_sim(config, simTime, input_current):
    """
    Simulate a Leaky Integrate-and-Fire (LIF) neuron.

    Args:
        config (dict): Configuration parameters.
        simTime (float): Simulation time in milliseconds.
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
    input_current = input_current
    simTime = simTime / 1000
    last_spike_time = 0
    membrane_voltage = [spike]
    i = 1

    # Perform simulation using Euler's method
    for time in np.linspace(0, simTime, int(simTime / delta_t)):
        # Check if the neuron is in the refractory period
        if time - last_spike_time <= refractory_period:
            voltage = rest_potential
        else:
            voltage = lifFunction(membrane_voltage[-1], input_current)

            if voltage >= spike_threshold:
                last_spike_time = time
                print(i)
                i = i + 1
                voltage = spike

        membrane_voltage.append(voltage)


    return membrane_voltage


def generate_spikes(spike_rate, run_time):
    # Calculate the expected number of spikes
    expected_spikes = spike_rate * run_time/1000


    # Discard spikes beyond the run time
    spike_times = np.arange(0, run_time/1000 + .01, expected_spikes/1000)

    return spike_times

def alpha_synapse_sim(config, simTime, spikeRate):
    """
    Simulate an alpha synapse.

    Args:
        config (dict): Configuration parameters.
        simTime (float): Simulation time in milliseconds.
        spike_times (list): List of spike times.

    Returns:
        ndarray: Array of synaptic currents over time.
    """
    # Extract configuration parameters
    reversal_potential = config["v_rev"]
    rest_potential = config["v_r"]
    spike_threshold = config["v_thr"]
    weight = config["w"]
    spike = config["v_spike"]
    decay_time = config["tao_syn"]
    delta_t = config["dt"]
    gBar = config["g_bar"]
    refractory_period = config["t_r"]

    # Initialize synaptic current array
    membrane_voltage = [rest_potential]
    last_input_spike_time = 0
    last_spike_time = -10000000

    spike_times = generate_spikes(spikeRate, simTime)
    i = 0

    for time in np.arange(0, simTime/1000, delta_t):

        timeFunc = (time - last_input_spike_time) / decay_time
        Isyn = weight * gBar * (reversal_potential - membrane_voltage[-1]) * timeFunc * math.exp(-timeFunc)

        if time - last_spike_time <= refractory_period:
            voltage = rest_potential

        else:
            if np.round(time, 4) != spike_times[i]:
                voltage = lifFunction(membrane_voltage[-1], Isyn)
            else:
                i = i + 1
                last_input_spike_time = time
                voltage = lifFunction(membrane_voltage[-1], Isyn)

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
        spikeRate = args.spike_rate
        synaptic_current = alpha_synapse_sim(config, args.simTime, spikeRate)
    else:  # args.mode == 'current'
        if args.current is None:
            parser.error('--current is required for "current" mode')
        input_current = args.current
        membrane_voltage = lif_neuron_sim(config, args.simTime, input_current)

    if args.mode == 'spike':
        plt.plot(synaptic_current)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Title')
        plt.grid(True)
        plt.show()
    else:  # args.mode == 'current'
        plt.plot(membrane_voltage)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Title')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()