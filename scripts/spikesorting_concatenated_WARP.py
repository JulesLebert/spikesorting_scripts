# Set logging before the rest as neo (and neo-based imports) needs to be imported after logging has been set
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('multirec_sorting')
logger.setLevel(logging.DEBUG)

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from jsmin import jsmin
import datetime
import re

import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.exporters as sexp

from spikeinterface import concatenate_recordings

from probeinterface import generate_multi_columns_probe

def generate_warp_16ch_probe():
    probe = generate_multi_columns_probe(num_columns=8,
                                        num_contact_per_column=2,
                                        xpitch=350, ypitch=350,
                                        contact_shapes='circle')
    probe.create_auto_shape('rect')

    channel_indices = np.array([13, 15,
                                9, 11,
                                14, 16,
                                10, 12,
                                8, 6,
                                4, 2,
                                7, 5,
                                3, 1])

    probe.set_device_channel_indices(channel_indices - 1)

    return probe

def preprocess_rec(recording):
    probe = generate_warp_16ch_probe()
    recording = recording.set_probe(probe)
    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')

    return recording_cmr


def export_all(working_directory, output_folder, job_kwargs):
    logger.info(f'saving {outDir} as phy')

    sorting_output = ss.collect_sorting_outputs(working_directory)
    for (rec_name, sorter_name), sorting in sorting_output.items():
        outDir = output_folder / rec_name / sorter_name
        we = sc.extract_waveforms(sorting._recording,
                                sorting, outDir / 'waveforms', 
                                ms_before=2.5, ms_after=3, 
                                max_spikes_per_unit=300, load_if_exists=True,
                                overwrite=False,
                                **job_kwargs
                                # n_jobs=10, 
                                # chunk_size=30000
                            )
        logger.info(f'WaveformExtractor: {we}')

        sexp.export_to_phy(we, outDir / 'phy', remove_if_exists=False, max_channels_per_template=3,
                copy_binary=True,
                **job_kwargs
                )
        logger.info(f'saved {outDir} as phy')
        sexp.export_report(we, outDir / 'report', 
                format='png',
                force_computation=True,
                **job_kwargs)
                
        logger.info(f'saving report')

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("params_file", help="path to the json file containing the parameters")
    # args = parser.parse_args()
    params_file = '/home/skgtjml/code/spikesorting_scripts/scripts/json_files/spikesorting_params_concatenated_WARP.json'
    with open(params_file) as json_file:
        minified = jsmin(json_file.read()) # Parses out comments.
        params = json.loads(minified)

    logpath = Path(params['logpath'])
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    fh = logging.FileHandler(logpath / f'multirec_warp_sorting_logs_{now}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Starting')

    sorter_list = params['sorter_list']
    logger.info(f'sorter list: {sorter_list}')

    if 'kilosort2' in sorter_list:
        ss.Kilosort2Sorter.set_kilosort2_path(params['sorter_paths']['kilosort2_path'])
    if 'waveclus' in sorter_list:
        ss.WaveClusSorter.set_waveclus_path(params['sorter_paths']['waveclus_path'])
    if 'kilosort3' in sorter_list:
        ss.Kilosort3Sorter.set_kilosort3_path(params['sorter_paths']['kilosort3_path'])

    datadir = Path(params['datadir']) / params['rec_name']

    streams = params['streams']

    output_folder = Path(params['output_folder']) / params['rec_name']
    output_folder.mkdir(parents=True, exist_ok=True)

    working_directory = Path(params['working_directory']) / params['rec_name']
    working_directory.mkdir(parents=True, exist_ok=True)

    blocks = [bl.name for bl in datadir.glob('BlockNellie*')]
    blocks.sort(key=lambda f: int(re.sub('\D', '', f)))
    pbar = tqdm(blocks)

    recording_list = {stream: [] for stream in streams}

    for stream in streams:
        logger.info(f'Loading stream {stream}')
        for block in pbar:
            pbar.set_postfix_str(f'loading {block}')
            logger.info(f'Loading block {block}')
            tdx_file = list((datadir / block).glob('*.Tdx'))
            assert len(tdx_file) == 1
            tdx_file = tdx_file[0]
            try:
                rec = se.read_tdt(tdx_file, stream_name=stream)
                rec = preprocess_rec(rec)
                recording_list[stream].append(rec)
            except:
                logger.info(f'Could not load block {block}')
 
    logger.info('Concatenating recordings')
    recordings = {stream: concatenate_recordings(recording_list[stream]) for stream in streams}

    logger.info('Sorting')
    for stream in streams:
        logger.info(f'Starting sorting for stream {stream}')
        sorting = ss.run_sorters(sorter_list, [recordings[stream]], working_folder=working_directory / stream, engine='loop', verbose=True)
        logger.info(f'Finished sorting for stream {stream}')

        export_all(working_directory=working_directory / stream, 
                output_folder=output_folder / stream,
                jobs_kwarg=params['jobs_kwargs']
                )

if __name__ == '__main__':
    main()