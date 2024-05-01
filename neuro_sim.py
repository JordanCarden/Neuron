import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def Count_Decimal_Places(number):
    num_str = str(number)

    if "." in num_str:
        integer_part, decimal_part = num_str.split(".")
        decimal_places = len(decimal_part.rstrip("0"))
        return decimal_places
    else:
        return 0


def Parse_Arguments():
    parser = argparse.ArgumentParser(description="Simulate LIF neuron model and alpha synapse model.")
    parser.add_argument("mode", choices=["spike", "current"], help="Simulation mode: 'spike' or 'current'")
    parser.add_argument("simTime", type=float, help="Simulation time in milliseconds")
    parser.add_argument("--spike_rate", type=int, help="Input spike rate in Hz (only for 'spike' mode)")
    parser.add_argument("--current", type=float, help="Input current in nanoamps (only for 'current' mode)")
    return parser.parse_args()


def Load_Config():
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    return config


def LIF_Function(previousVoltage, inputCurrent):
    config = Load_Config()
    restPotential = config["v_r"]
    deltaTime = config["dt"]
    decayTime = config["tao_m"]
    membraneCapacity = config["c_m"]

    return previousVoltage + deltaTime * (((-previousVoltage + restPotential) / decayTime) + (inputCurrent/(membraneCapacity)))


def LIF_Neuron_Model(config, simTime, inputCurrent):
    restPotential = config["v_r"]
    spikeThreshold = config["v_thr"]
    spikeVoltage = config["v_spike"]
    refractoryPeriod = config["t_r"]
    deltaTime = config["dt"]

    inputCurrent = inputCurrent/1000000000
    lastSpikeTime = -1000000
    membraneVoltage = [restPotential]
    deltaTime = deltaTime * 1000
    refractoryPeriod = refractoryPeriod * 1000

    for time in np.arange(0, simTime, deltaTime):
        time = np.round(time, Count_Decimal_Places(deltaTime))

        if time - lastSpikeTime <= refractoryPeriod:
            voltage = restPotential
        else:
            voltage = LIF_Function(membraneVoltage[-1], inputCurrent)

            if voltage >= spikeThreshold:
                lastSpikeTime = time
                voltage = spikeVoltage

        membraneVoltage.append(voltage)

    return membraneVoltage


def Taylor_Series_Expansion(x, order=10):

    approximation = 1
    term = 1

    for n in range(1, order + 1):
        term *= x / n
        approximation += term

    return approximation


def Generate_Spikes(spikeRate, run_time):
    expectedSpikes = spikeRate * run_time/1000
    spikeTimes = np.linspace(0, run_time, int(expectedSpikes+1))
    print(spikeTimes)
    return spikeTimes


def Alpha_Synapse_Model(config, simTime, spikeRate):
    reversal_potential = config["v_rev"]
    restPotential = config["v_r"]
    spikeThreshold = config["v_thr"]
    weight = config["w"]
    spikeVoltage = config["v_spike"]
    decayTime = config["tao_syn"]
    deltaTime = config["dt"]
    gBar = config["g_bar"]
    refractoryPeriod = config["t_r"]

    membraneVoltage = [restPotential]
    lastInputSpikeTime = 0
    lastSpikeTime = -1000000
    deltaTime = deltaTime * 1000
    refractoryPeriod = refractoryPeriod * 1000
    decayTime = decayTime * 1000

    spikeTimes = Generate_Spikes(spikeRate, simTime)
    i = 0

    for time in np.arange(0, simTime, deltaTime):
        time = np.round(time, Count_Decimal_Places(deltaTime))
        timeFunc = (time - lastInputSpikeTime) / decayTime
        #expApproximation = taylorSeriesExpansion(-timeFunc)
        expApproximation = np.exp(-timeFunc)
        Isyn = weight * gBar * (reversal_potential - membraneVoltage[-1]) * timeFunc * expApproximation

        if time - lastSpikeTime <= refractoryPeriod:
            voltage = restPotential

        else:
            if time != spikeTimes[i]:
                voltage = LIF_Function(membraneVoltage[-1], Isyn)
            else:
                i = i + 1
                lastInputSpikeTime = time
                voltage = LIF_Function(membraneVoltage[-1], Isyn)

            if voltage >= spikeThreshold:
                lastSpikeTime = time
                voltage = spikeVoltage

        membraneVoltage.append(voltage)

    return membraneVoltage


def main():
    args = Parse_Arguments()
    config = Load_Config()

    if args.mode == 'spike':
        if args.spike_rate is None:
            parser.error('--spike_rate is required for "spike" mode')
        spikeRate = args.spike_rate
        synapticCurrent = Alpha_Synapse_Model(config, args.simTime, spikeRate)
    else:
        if args.current is None:
            parser.error('--current is required for "current" mode')
        inputCurrent = args.current
        membraneVoltage = LIF_Neuron_Model(config, args.simTime, inputCurrent)

    if args.mode == 'spike':
        plt.plot(np.linspace(0, args.simTime, len(synapticCurrent)), synapticCurrent)
        plt.xlabel('Time (msec)')
        plt.ylabel('Membrane Potential (volt)')
        plt.title('Voltage Track Group 12')
        plt.grid(False)
        plt.show()
    else:
        plt.plot(np.linspace(0, args.simTime, len(membraneVoltage)), membraneVoltage)
        plt.xlabel('Time (msec)')
        plt.ylabel(r'$V_{m}$ (volt)')
        plt.title('Membrane Potential Track Group 12')
        plt.grid(False)
        plt.show()

if __name__ == "__main__":
    main()
