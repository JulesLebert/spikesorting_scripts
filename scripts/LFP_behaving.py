from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from tqdm import tqdm

import tdt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from probeinterface.plotting import plot_probe

from instruments.helpers.extract_helpers import load_bhv_file

from spikesorting_scripts.npyx_metadata_fct import get_npix_sync





def plot_trig_to_pulse_time(
        df_bhv,
        trig,
        trig_fs,
        n_pulses=10
        ):
    df_bhv_subset = df_bhv.sample(n=n_pulses, random_state=1)

    fig, axes = plt.subplots(n, 1, figsize=(7, 10), sharex=True, sharey=True)

    window = np.array([-.3, .3])
    window_samples = (window * trig_fs).astype(int)

    for ax, pulse, start_lick in zip(axes, df_bhv_subset.npPulseTime, df_bhv_subset.startTrialLick):
        pulses_samples = (pulse * trig_fs).astype(int)
        lick_samples = (start_lick * trig_fs).astype(int)
        times = np.arange(window_samples[0], window_samples[1]) / trig_fs
        ax.plot(times, trig[1, pulses_samples + window_samples[0]: pulses_samples + window_samples[1]])
        ax.axvline(0, color='r', label='pulse')
        ax.axvline((lick_samples - pulses_samples) / trig_fs, color='g', label='lick')
        ax.legend()

    fig.tight_layout()


def get_lfp_to_pulses(
        recording_lfp,
        df_bhv,
        window,
        channel_ids,
        ):
    
    lfp_fs = recording_lfp.get_sampling_frequency()
    pulse_time = df_bhv.imec_pulse_time

    window_samples = (window * lfp_fs).astype(int)
    pulses_samples = (pulse_time * lfp_fs).astype(int)

    traces = np.zeros((len(pulses_samples), np.diff(window_samples)[0], len(channel_ids)))
    
    for i, pulse in tqdm(enumerate(pulses_samples), desc='loading aligned traces', total=len(pulses_samples)):
    #     rel_lick_time = (lick_samples - pulses_samples) / trig_fs
        pulse_traces = recording_lfp.get_traces(
                start_frame=pulse+window_samples[0],
                end_frame=pulse+window_samples[1],
                channel_ids=channel_ids
        )
        traces[i,:,:] = pulse_traces[:traces.shape[1],:]

    return traces

def plot_lfp_to_pulses(
        recording_lfp,
        df_bhv,
        ):
    
    lfp_fs = recording_lfp.get_sampling_frequency()
    window = np.array([-.5, .5])

    channel_indices = np.arange(0,385,20)
    channel_ids = recording_lfp.channel_ids[channel_indices]
    
    traces = get_lfp_to_pulses(recording_lfp,
                df_bhv,
                window,
                channel_ids,
                )

    fig, ax = plt.subplots(figsize = (10,5), dpi=300)
    times = np.arange(window[0], window[1], 1/lfp_fs)
    for chan in range(traces.shape[2]):
        to_plot = np.mean(np.squeeze(traces[:,:,chan]), axis=0)
        if not np.max(to_plot) > 50:
            ax.plot(
                times,
                to_plot,
                label=channel_ids[chan],
                )
        
    ax.axvline(0, linestyle='--', linewidth=2)
    # ax.legend()
    ax.set_xlabel('Time from trial onset (s)')
    ax.set_title('Orecchiette')

    return fig


def get_lfp_to_lick(
        recording_lfp,
        df_bhv,
        window,
        channel_ids,
    ):
    lfp_fs = recording_lfp.get_sampling_frequency()

    window_samples = (window * lfp_fs).astype(int)
    # pulses_samples = (pulse_time * lfp_fs).astype(int)

    lick_times = df_bhv.lick_imec_time.values
    lick_samples = (lick_times * lfp_fs).astype(int)


    traces = np.zeros((len(lick_samples), np.diff(window_samples)[0], len(channel_ids)))

    for i, pulse in tqdm(enumerate(lick_samples), desc='loading aligned traces', total=len(lick_samples)):
    #     rel_lick_time = (lick_samples - pulses_samples) / trig_fs
        pulse_traces = recording_lfp.get_traces(
                start_frame=pulse+window_samples[0],
                end_frame=pulse+window_samples[1],
                channel_ids=channel_ids
        )
        traces[i,:,:] = pulse_traces[:traces.shape[1],:]

    return traces


def plot_lfp_to_lick(
        recording_lfp,
        df_bhv,
        ):
    window = np.array([-.5, 1])

    channel_indices = np.arange(0,384,20)
    channel_ids = recording_lfp.channel_ids[channel_indices]

    traces = get_lfp_to_lick(recording_lfp,
                        df_bhv,
                        window,
                        channel_ids,
                        )

    lfp_fs = recording_lfp.get_sampling_frequency()

    fig, ax = plt.subplots(figsize = (10,5), dpi=300)
    times = np.arange(window[0], window[1], 1/lfp_fs)
    for chan in range(traces.shape[2]):
        to_plot = np.mean(np.squeeze(traces[:,:,chan]), axis=0)
        # if not np.max(to_plot) > 50:
        ax.plot(
            times,
            to_plot,
            label=channel_ids[chan],
            )
    
    ax.axvline(0, linestyle='--', linewidth=2)
    # ax.legend()
    ax.set_xlabel('Time from trial onset (s)')
    ax.set_title(' ')

    fig.tight_layout()
    # fig.show()

    return fig


def generate_pdf_single_channels(
        recording_lfp,
        df_bhv,
        savedir,
        title,
        align_to = 'licks'
        ):
    
    assert align_to in ['licks', 'pulses']

    window = np.array([-.5, 1])

    channel_indices = np.arange(0,384,20)
    channel_ids = recording_lfp.channel_ids[channel_indices]
    lfp_fs = recording_lfp.get_sampling_frequency()

    if align_to == 'licks':
        traces = get_lfp_to_lick(
            recording_lfp,
            df_bhv,
            window,
            channel_ids,
        )
    else:
        traces = get_lfp_to_pulses(
            recording_lfp,
            df_bhv,
            window,
            channel_ids,
        )

    savedir = Path(savedir)
    with PdfPages(savedir / f'{title}.pdf') as pdf:
        print(f'saving LFP to {savedir}/{title}.pdf')
        times = np.arange(window[0], window[1], 1/lfp_fs)
        for chan in tqdm(range(traces.shape[2]), desc='Plotting channels'):
            fig, axes = plt.subplots(1, 2, figsize = (10,5), dpi=300)
            pos = axes[1].imshow(
                    np.squeeze(traces[:,:,chan]),
                    extent=[times[0], times[-1], 0, traces.shape[0]],
                    cmap='magma', 
                    aspect='auto',
                    )

            plot_probe_curr_channel(
                recording_lfp,
                chan,
                channel_indices,
                axes[0],
            )

            fig.colorbar(pos, ax=axes[0])


            fig.suptitle(f'LFP for {channel_ids[chan]}')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def plot_probe_curr_channel(
                recording,
                chan,
                channel_indices,
                ax,
            ):
    probe = recording.get_probe()

    chan_index = channel_indices[chan]
    chan_id = recording.channel_ids[chan_index]
    loc_y = recording.get_channel_locations([chan_id])[0][1]

    values = np.zeros(len(probe.device_channel_indices))
    values[int(chan_index)] = 1
    plot_probe(probe, with_channel_index=True,
        contacts_values = values, ax=ax)
    
    ax.set_ylim(loc_y-150, loc_y+150)

    return ax



def main():
    npx_data = Path('/mnt/a/NeuropixelData/raw') #/F2003_Orecchiette/090523_ore_am_g0')
    bhv_path = Path('/home/jules/Dropbox/Data') # /F2003_Orecchiette')

    # dp = npx_data / 'F2003_Orecchiette' / '090523_ore_am_g0'

    dp = npx_data / 'F2003_Orecchiette' / '15112022_Ore_AM_g0'


    # dp = npx_data / 'F1903_Trifle_1' / '251022_Trifle_AM_g0'

    # dp = npx_data / 'F1903_Trifle_1' / '311022_Trifle_AM_g0'

    # dp = npx_data / 'F1903_Trifle_1' / '271022_Trifle_AM_g0'


    # bhv_file = bhv_path / 'F2003_Orecchiette' / '9_5_2023 level48_Ore15SSN.txt 11_43.mat'
    bhv_file = bhv_path / 'F2003_Orecchiette' / '15_11_2022 level41_Orecchiette_passive.txt 11_53.mat'

    # bhv_file = bhv_path / 'F1903_Trifle' / '25_10_2022 level41_Trifle15SSN.txt 13_6.mat'
    # bhv_file = bhv_path / 'F1903_Trifle' / '31_10_2022 level41_TrifleNoNoiseprobes.txt 12_33.mat'
    # bhv_file = bhv_path / 'F1903_Trifle' / '27_10_2022 level41_Trifle15SSN.txt 12_9.mat'

    rec = se.read_spikeglx(dp, stream_id='imec0.ap')
    recording_lfp = spre.bandpass_filter(rec, freq_min=1, freq_max=45)
    recording_lfp = spre.resample(recording_lfp, 1000)

    pulse_time, _ = get_npix_sync(dp, output_binary=False, unit='seconds', sync_trial_chan=[5], verbose=True)
    pulse_time = pulse_time[5]

    # load behavior
    df_bhv = load_bhv_file(str(bhv_file))

    imec_pulse_time = pulse_time[:len(df_bhv.npPulseTime)]
    assert np.allclose(np.diff(df_bhv.npPulseTime), np.diff(imec_pulse_time), atol=0.01), \
        f'Behavioural file and sync pulses data do not match (probably because of a missing pulse somewhere)'
    df_bhv['imec_pulse_time'] = imec_pulse_time
    df_bhv['lick_imec_time'] = df_bhv.startTrialLick - df_bhv.npPulseTime + df_bhv.imec_pulse_time

    # blockpath = Path('/mnt/b/WarpData/behaving/raw') #/F2003_Orecchiette')
    # tdt_block = tdt.read_block(blockpath / 'F1903_Trifle' / 'BlockNellie-AA-94') # 'BlockNellie-AB-4')

    # trig = tdt_block.streams.trig.data
    # trig_fs = tdt_block.streams.trig.fs

    fig = plot_lfp_to_lick(recording_lfp, df_bhv)
    # fig = plot_lfp_to_pulses(recording_lfp, df_bhv)

    savedir = Path('/home/jules/figures/LFP')
    # # figname = 'LFP_Trifle_311022_2s_5_45hz.png'
    # figname = 'LFP_Orecchiette_45Hz_05s.png'

    figname = f'mean_LFP {dp.name}.png'
    fig.savefig(savedir / figname, dpi=300)

    title = f'LFP per channel {dp.name}'
    generate_pdf_single_channels(
        recording_lfp,
        df_bhv,
        savedir=savedir,
        title=title,
        align_to='licks'
    )



if __name__ == "__main__":
    main()


TDT_FS = 24414.062500