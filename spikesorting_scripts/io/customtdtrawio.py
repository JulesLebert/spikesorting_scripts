"""
ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits from BaseRawIO
    * copy/paste all methods that need to be implemented.
    * code hard! The main difficulty is `_parse_header()`.
      In short you have a create a mandatory dict than
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_streams'] = signal_streams
            self.header['signal_channels'] = signal_channels
            self.header['spike_channels'] = spike_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py



"""

from neo.rawio.baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from pathlib import Path
import tdt


class CustomTdtRawIO(BaseRawIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it gives access to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`

    This fake IO:
        * has 2 blocks
        * blocks have 2 and 3 segments
        * has  2 signals streams  of 8 channel each (sample_rate = 10000) so 16 channels in total
        * has 3 spike_channels
        * has 2 event channels: one has *type=event*, the other has
          *type=epoch*


    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.ExampleRawIO(filename='itisafake.nof')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
        >>> spike_timestamp = reader.spike_timestamps(spike_channel_index=0,
                            t_start=None, t_stop=None)
        >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname='',store=None):
        BaseRawIO.__init__(self)
        # note that this filename is ued in self._source_name
        self.dirname = dirname
        if store==None:
            store = ['BB_2','BB_3','BB_4','BB_5']
        self.store=store

    def _source_name(self):
        # this function is used by __repr__
        # for general cases self.filename is good
        # But for URL you could mask some part of the URL to keep
        # the main part.
        return self.dirname

    def _parse_header(self):

        tdt_folder = Path(self.dirname)
        self._tdt_data = tdt.read_block(tdt_folder,store=self.store)

        # signal_streams = []
        # signal_channels = []
        # chan_nb = 0
        # for s, stream in enumerate(self._tdt_data.streams.keys()):
        #     if stream.startswith('BB'):
        #         name = stream
        #         stream_id = s
        #         signal_streams.append((name, stream_id))

        #         for c, channel in enumerate(self._tdt_data.streams[stream].channels):
        #             ch_name = f'{channel}'
        #             chan_id = channel-1
        #             fs = self._tdt_data.streams[stream].fs
        #             dtype = 'int16'
        #             units = 'uV'
        #             gain = 1
        #             offset = 0

        #             signal_channels.append((ch_name, chan_id, fs, dtype, units,
        #                         gain, offset, stream_id))
        #             chan_nb+=1

        signal_channels = []
        chan_nb = 0
        stream_id = 0
        signal_streams = [('all',stream_id)]

        chan_nb = 0

        for s, stream in enumerate(self._tdt_data.streams.keys()):
            if stream.startswith('BB'):
                for c, channel in enumerate(self._tdt_data.streams[stream].channels):
                    ch_name = f'{channel}_{stream}'
                    chan_id = chan_nb
                    fs = self._tdt_data.streams[stream].fs
                    dtype = 'int16'
                    units = 'uV'
                    gain = 1
                    offset = 0

                    signal_channels.append((ch_name, chan_id, fs, dtype, units,
                                gain, offset, stream_id))
                    chan_nb+=1


        
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)


        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['name'] = "Block #0"
        seg_ann = bl_ann['segments'][0]
        seg_ann['name'] = self._tdt_data.info.blockname
        seg_ann['ferret'] = tdt_folder.parents._parts[-2]

        self._t_start = 0
        self._t_stop = self._tdt_data.info.duration.total_seconds()




    def _segment_t_start(self, block_index, seg_index):
        assert block_index == 0
        return self._t_start

    def _segment_t_stop(self, block_index, seg_index):
        assert block_index == 0
        return self._t_stop[0]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        # We generate fake data in which the two stream signals have the same shape
        # across all segments (10.0 seconds)
        # This is not the case for real data, instead you should return the signal
        # size depending on the block_index and segment_index
        # this must return an int = the number of sample

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.
        assert block_index == 0
        assert seg_index == 0

        size = self._tdt_data.streams[self.store[0]].data.shape[1]
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        # This give the t_start of signals.
        # Very often this equal to _segment_t_start but not
        # always.
        # this must return an float scale in second

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.

        # Here this is the same.
        # this is not always the case
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        # this must return a signal chunk in a signal stream
        # limited with i_start/i_stop (can be None)
        # channel_indexes can be None (=all channel in the stream) or a list or numpy.array
        # This must return a numpy array 2D (even with one channel).
        # This must return the orignal dtype. No conversion here.
        # This must as fast as possible.
        # To speed up this call all preparatory calculations should be implemented
        # in _parse_header().

        # Here we are lucky:  our signals is always zeros!!
        # it is not always the case :)
        # internally signals are int16
        # convertion to real units is done with self.header['signal_channels']
        
        # stream_name = [i[0] for i in self.header['signal_streams'] if i[1]==f'{stream_index}'][0]

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._tdt_data.streams[self.store[0]].data.shape[1]

        if i_start < 0 or i_stop > self._tdt_data.streams[self.store[0]].data.shape[1]:
            # some check
            raise IndexError("I don't like your jokes")
        
        if channel_indexes is None:
            # If none, load all channels
            channel_indexes = np.arange(np.sum([self._tdt_data.streams[store].data.shape[0] for store in self.store]))


        # raw_signals = self._tdt_data.streams[stream_name].data[channel_indexes,:].T
        raw_signals = np.vstack([self._tdt_data.streams[store].data for store in self.store])
        
        return raw_signals[channel_indexes,i_start:i_stop].T

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        # Must return the nb of spikes for given (block_index, seg_index, spike_channel_index)
        # we are lucky:  our units have all the same nb of spikes!!
        # it is not always the case
        return None

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index,
                                 t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return None