import numpy as np
from scipy.signal import welch, find_peaks

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

from spikeinterface.core import get_random_data_chunks

class RemoveDisconnectionEventRecording(BasePreprocessor):
    """
    Preprocessor to remove disconnection events from a recording.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to preprocess.
    fill_value : float or None, optional
        Value to fill disconnection events with. If None, the median value of the recording is used.
    compute_medians : str, optional
        If 'random', compute the median power spectrum of a subset of the recording's data. If 'all', compute the median
        power spectrum of the entire recording's data. Defaults to 'random'.
    n_peaks : int, optional
        The minimum number of peaks in the power spectrum required to compute the median power spectrum.
        Defaults to 10.
    prominence : float, optional
        The prominence parameter used to find peaks in the power spectrum. Defaults to 0.5.
    n_median_threshold : int, optional
        The number of standard deviations from the median power spectrum above which a data segment is considered a
        disconnection event. Defaults to 2.
    num_chunks_per_segment : int, optional
        The number of data chunks to use in each segment when computing the median power spectrum. Defaults to 100.
    chunk_size : int, optional
        The number of samples per chunk when computing the median power spectrum. Defaults to 10000.
    seed : int, optional
        Seed for the random number generator when computing the median power spectrum with random chunks. Defaults to 0.

    Raises
    ------
    AssertionError
        If `compute_medians` is not one of ['random', 'all'].

    Notes
    -----
    This preprocessor first computes the median power spectrum of the recording's data by either randomly selecting a subset
    of the data or using all the data. Then, it identifies disconnection events by comparing the power spectra of each data
    segment to the median power spectrum. Data segments with power spectra more than `n_median_threshold` standard
    deviations above the median are considered disconnection events and are filled with the value specified by `fill_value`
    (or the median value of the recording if `fill_value` is None).

        """
    
    name='remove_disconnection_events'

    def __init__(self, recording,
                 fill_value=None, compute_medians="random",
                 n_peaks=10, prominence=0.5, n_median_threshold=2,
                 num_chunks_per_segment=100, chunk_size=10000, 
                 seed=0
                 ):
        """
        Remove disconnection event from recording.
        """

        assert compute_medians in ['random', 'all']
        if compute_medians == 'random':
            subset_data = get_random_data_chunks(recording,
                                                num_chunks_per_segment=num_chunks_per_segment,
                                                chunk_size=chunk_size, seed=seed)
        elif compute_medians == 'all':
            subset_data = recording.get_traces()
        
        fs = recording.get_sampling_frequency()
        f, Pxx = welch(subset_data, fs, nperseg=1024, detrend=False, axis=0)
        Pxx_dB = np.mean(10 * np.log10(Pxx), axis=1)
        peaks, _ = find_peaks(Pxx_dB, prominence=prominence)

        if len(peaks) < n_peaks:
            median_power = None
        else:
            if compute_medians == 'all':
                num_chunks_per_segment = subset_data.shape[0] // chunk_size
                subset_data = subset_data[:num_chunks_per_segment*chunk_size, :]

            subset_data_reshaped = subset_data.reshape((num_chunks_per_segment, chunk_size, subset_data.shape[-1]))

            # power = np.sum(np.abs(random_data_reshaped)**2, axis=1)/random_data_reshaped.shape[1]
            # power = np.mean(np.square(np.abs(random_data_reshaped)), axis=1)
            power = np.mean(np.abs(subset_data_reshaped), axis=1)
            median_power = np.median(power, axis=0)

            if fill_value is None:
                fill_value = np.median(subset_data_reshaped)
        
        BasePreprocessor.__init__(self, recording)
        for parent_segment in recording._recording_segments:
            rec_segment = RemoveDisconnectionEventRecordingSegment(
                parent_segment, median_power, n_median_threshold,
                fill_value=fill_value, chunk_size=chunk_size,
                )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(recording=recording, n_peaks=n_peaks, prominence=prominence, fill_value=fill_value,
                            n_median_threshold=n_median_threshold, num_chunks_per_segment=num_chunks_per_segment, 
                            chunk_size=chunk_size, seed=seed,
                        )

class RemoveDisconnectionEventRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_recording_segment,
                 median_power, n_median_threshold,
                 fill_value, chunk_size=10000,
                 ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.median_power = median_power
        self.n_median_threshold = n_median_threshold
        self.fill_value = fill_value
        self.chunk_size = chunk_size

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        if self.median_power is None:
            return traces
        else:
            traces = traces.copy()
            median_power = self.median_power[channel_indices]

            # chunk_powers = []
            for i in range(0, traces.shape[0], self.chunk_size):
                chunk = traces[i:i+self.chunk_size, :]
                chunk_power = np.mean(np.square(np.abs(chunk)), axis=0)
                chunk_power = np.mean(np.abs(chunk), axis=0)
                # chunk_powers.append(chunk_power)
                # chunk_power = np.sum(np.abs(x)**2)/len(x)
                mask = np.greater(chunk_power, self.n_median_threshold*median_power)

                chunk[:, mask] = self.fill_value
                traces[i:i+self.chunk_size, :] = chunk
        
        # chunk_powers = np.vstack(chunk_powers)

        return traces
    
remove_disconnection_events = define_function_from_class(RemoveDisconnectionEventRecording, 
                                                        name="remove_disconnection_event")
