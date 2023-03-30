import numpy as np
from pathlib import Path

def pad_amplitude(spike_times, amplitudes):
    padded_amplitudes = np.zeros(len(spike_times))
    padded_amplitudes[:len(amplitudes)] = amplitudes
    return amplitudes

def save_padded_amplitudes(phy_folder):
    spike_times = np.load(phy_folder / 'spike_times.npy')
    amplitudes = np.load(phy_folder / 'amplitudes.npy')
    padded_amplitudes = pad_amplitude(spike_times, amplitudes)
    np.save(phy_folder / 'original_amplitudes.npy', amplitudes)
    np.save(phy_folder / 'amplitudes.npy', padded_amplitudes)

def main():
    phy_folder = Path('/home/skgtjml/Scratch/si_neuropixel_spikesorting/081222_fet_pm_g0/pykilosort/phy_folder')
    save_padded_amplitudes(phy_folder)

if __name__ == '__main__':
    main()