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
from dataclasses import dataclass
import datetime
import re

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.exporters import export_to_phy, export_report

import spikeinterface.preprocessing as spre
import spikeinterface.core as sc
import spikeinterface.exporters as sexp

from spikeinterface import concatenate_recordings

from probeinterface import generate_multi_columns_probe

from spikesorting_scripts.io.customtdtrecording_extractor import CustomTdtRecordingExtractor

@dataclass
class TDTData:
    dp: str
    store: list

def generate_warp_32ch_probe():
    probe = generate_multi_columns_probe(num_columns=8,
                                         num_contact_per_column=4,
                                         xpitch=350, ypitch=350,
                                         contact_shapes='circle')
    probe.create_auto_shape('rect')

    channel_indices = np.array([29, 31, 13, 15,
                                25, 27, 9, 11,
                                30, 32, 14, 16,
                                26, 28, 10, 12,
                                24, 22, 8, 6,
                                20, 18, 4, 2,
                                23, 21, 7, 5,
                                19, 17, 3, 1])

    probe.set_device_channel_indices(channel_indices - 1)

    return probe


def preprocess_data(data):
    recording = CustomTdtRecordingExtractor(data.dp, store=data.store)

    probe = generate_warp_32ch_probe()
    recording = recording.set_probe(probe)

    recording_f = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')

    # self.recording_preprocessed = recording_cmr
    return recording_cmr



def run_ks2_cg(data, output_folder):
    data.sorting_KS = ss.run_kilosort2(recording=data,
                                       output_folder=output_folder)

    print(data.sorting_KS)

def export_all_as_phy(working_directory, output_folder, store=['BB_4','BB_5']):
    # Export a working directory as phy

    job_kwargs = dict(chunk_duration='1s', n_jobs=6)

    sorting_output = ss.collect_sorting_outputs(working_directory)
    for (rec_name, sorter_name), sorting in sorting_output.items():
        try:
            outDir = output_folder / rec_name / sorter_name
            # data = TDTData(datadir, rec_name, store)
            # data.load_tdtRec()
            # # data.load_multiple_tdtRec()
            # data.preprocess_data()
            if not outDir.is_dir():
                logger.info(f'saving {outDir} as phy')

                we = si.extract_waveforms(sorting._recording,
                                            sorting, outDir / 'waveforms', 
                                            ms_before=2.5, ms_after=3, 
                                            max_spikes_per_unit=4000, load_if_exists=True,
                                            overwrite=False,
                                            **job_kwargs
                                            # n_jobs=10, 
                                            # chunk_size=30000
                                        )

                logger.info(f'WaveformExtractor: {we}')
                export_to_phy(we, outDir / 'phy', remove_if_exists=False, max_channels_per_template=3,
                                chunk_size=30000,
                                copy_binary=False,
                                **job_kwargs
                                # n_jobs=3,
                                )
                logger.info(f'saved {outDir} as phy')
                export_report(we, outDir / 'report', **job_kwargs)
                logger.info(f'saving report')
            else:
                logger.info(f'{outDir} already exists')
        except Exception as e:
            logger.info(f'Failed to export as phy. Error: {e}')

def export_all(working_directory, output_folder, job_kwargs):

    sorting_output = ss.collect_sorting_outputs(working_directory)
    for (rec_name, sorter_name), sorting in sorting_output.items():
        outDir = output_folder / rec_name / sorter_name
        logger.info(f'saving {outDir} as phy')
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

        sexp.export_to_phy(we, outDir / 'phy', remove_if_exists=True,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="path to the json file containing the parameters")
    args = parser.parse_args()
    # params_file = '/home/jules/code/WARPAutomatedSpikesorting/spikesorting_params.json'
    with open(args.params_file) as json_file:
        minified = jsmin(json_file.read()) # Parses out comments.
        params = json.loads(minified)

    logpath = Path(params['logpath'])
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    fh = logging.FileHandler(logpath / f'multirec_sorting_logs_{now}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Starting')

    sorter_list = params['sorter_list'] #['klusta'] #'kilosort2']
    logger.info(f'sorter list: {sorter_list}')

    if 'kilosort2' in sorter_list:
        ss.Kilosort2Sorter.set_kilosort2_path(params['sorter_paths']['kilosort2_path'])
    if 'waveclus' in sorter_list:
        ss.WaveClusSorter.set_waveclus_path(params['sorter_paths']['waveclus_path'])
    if 'kilosort3' in sorter_list:
        ss.Kilosort3Sorter.set_kilosort3_path(params['sorter_paths']['kilosort3_path'])

    datadir = Path(params['datadir']) / params['rec_name']

    store = params['streams']
 
    # csv_dir = '/home/jules/code/WARPAutomatedSpikesorting/outdir/F1903_Trifle_Block_infos.csv'
    # output_folder = Path(params['output_folder'])
    # working_directory = Path(params['working_directory']) / 'tempDir'

    output_folder = Path(params['output_folder']) / params['rec_name']
    output_folder.mkdir(parents=True, exist_ok=True)

    working_directory = Path(params['working_directory']) / params['rec_name']
    working_directory.mkdir(parents=True, exist_ok=True)

    recording_list = []

    blocks = [bl.name for bl in datadir.glob('BlockNellie*')]
    blocks.sort(key=lambda f: int(re.sub('\D', '', f)))
    pbar = tqdm(blocks)

    logger.info('Start loading blocks')

    for block in pbar:
        pbar.set_postfix_str(f'loading {block}')
        logger.info(f'loading {block}')

        try:
            data = TDTData(datadir / block, store)
            new_data = preprocess_data(data)

            # Try to load traces to check if size BB_4 = size BB_5
            traces = new_data.get_traces()


            recording_list.append(new_data)
        
        except Exception as e:
            logger.info(f'Failed to load {block}. Error: {e}')


        # recording = CustomTdtRecordingExtractor(block, store=['BB_4','BB_5'])
        # recording = recording.set_probe(probe)
        # recording_list.append(recording)

    logger.info('Start concatenating recordings')
    rec = concatenate_recordings(recording_list)
    logger.info(rec)
    s = rec.get_num_samples(segment_index=0)
    logger.info(f'segment {0} num_samples {s}')

    recordings = {params['rec_name']: rec}
    logger.info(f'start sorting with {sorter_list}')

    # data_sorted = ss.run_sorters(sorter_list, recordings, working_folder=working_directory,
    #         mode_if_folder_exists='keep')
    sorting = ss.run_sorters(sorter_list, recordings, working_folder=working_directory,
            engine='loop', verbose=True,
            mode_if_folder_exists='keep',
            sorter_params=params['sorter_params']
            )

    logger.info(f'sorting_done')

    logger.info(f'export_to_phy')

    export_all(working_directory=working_directory, 
            output_folder=output_folder,
            job_kwargs=params['job_kwargs']
            )

    # export_all_as_phy(working_directory, output_folder)

if __name__ == '__main__':
    main()