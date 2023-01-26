import os
import numpy as np

def pad_amplitude(spike_time, amplitudes):
    padded_amplitudes = np.zeros_like(spike_time)
    padded_amplitudes[:len(amplitudes)] = amplitudes
    return padded_amplitudes

def save_padded_amplitudes(phy_folder):
    spike_times = np.load(phy_folder / 'spike_times.npy')
    amplitudes = np.load(phy_folder / 'amplitudes.npy')
    padded_amplitudes = pad_amplitude(spike_times, amplitudes)
    np.save(phy_folder / 'original_amplitudes.npy', amplitudes)
    np.save(phy_folder / 'amplitudes.npy', padded_amplitudes)
    return

def rename_template_id(phy_folder):
    if (phy_folder / 'template_ind.npy').exists():
        os.rename(phy_folder / 'template_ind.npy', phy_folder / 'template_inds.npy')
    return

def postprocessing_si(phy_folder):
    save_padded_amplitudes(phy_folder)
    rename_template_id(phy_folder)