import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('sorting')
logger.setLevel(logging.DEBUG)

from pathlib import Path
import datetime
import argparse
import json
from jsmin import jsmin
import re

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.curation as scu
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp

from spikesorting_scripts.postprocessing import postprocessing_si
def spikeglx_preprocessing(recording):
    recording = spre.phase_shift(recording)
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = spre.common_reference(recording, reference='global', operator='median')
    return recording

def spikesorting_pipeline(rec_name, params):
    working_directory = Path(params['working_directory']) / 'tempDir'

    logger.info(f'preprocessing recording')
    recording = se.read_spikeglx(rec_name, stream_id = 'imec0.ap')
    recording = spikeglx_preprocessing(recording)

    logger.info(f'running spike sorting')
    sorting_output = ss.run_sorters(params['sorter_list'], [recording], working_folder=working_directory,
        mode_if_folder_exists='overwrite', 
        verbose=True, 
        engine='loop', 
        sorter_params=params['sorter_params']
        # engine_kwargs={'n_jobs': params['jobs_kwargs']['n_jobs']}
        )

def spikesorting_postprocessing(rec, params):
    jobs_kwargs = params['jobs_kwargs']
    sorting_output = ss.collect_sorting_outputs(Path(params['working_directory']) / 'tempDir')
    for (rec_name, sorter_name), sorting in sorting_output.items():
        logger.info(f'Postprocessing {rec_name} {sorter_name}')
        if params['remove_dup_spikes']:
            logger.info(f'removing duplicate spikes')
            sorting = scu.remove_duplicated_spikes(sorting, censored_period_ms=params['remove_dup_spikes_params']['censored_period_ms'])

        logger.info('waveform extraction')
        outDir = Path(params['output_folder']) / rec.name / sorter_name

        we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder',
                load_if_exists=True,
                overwrite=False,
                ms_before=1, ms_after=2., max_spikes_per_unit=100,
                **jobs_kwargs)

        logger.info(f'Exporting to phy')
        sexp.export_to_phy(we, outDir / 'phy_folder',
            verbose=True, 
            compute_pc_features=False,
            **jobs_kwargs)
        
        postprocessing_si(outDir / 'phy_folder')

        sorting = se.read_kilosort(outDir / 'phy_folder')

        we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder',
            load_if_exists=False,
            overwrite=True,
            ms_before=1, ms_after=2., max_spikes_per_unit=100,
            **jobs_kwargs)

        logger.info(f'Computing quality metrics')
        metrics = sqm.compute_quality_metrics(we, n_jobs = jobs_kwargs['n_jobs'], verbose=True)

        try:
            logger.info('Export report')
            sexp.export_report(we, outDir / 'report',
                    format='png',
                    force_computation=True,
                    **jobs_kwargs)
        except Exception as e:
            logger.warning(f'Export report failed: {e}')


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="path to the json file containing the parameters")
    args = parser.parse_args()
    with open(args.params_file) as json_file:
        minified = jsmin(json_file.read()) # Parses out comments.
        params = json.loads(minified)

    logpath = Path(params['logpath'])
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    fh = logging.FileHandler(logpath / f'neuropixels_sorting_logs_{now}.log')
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

    datadir = Path(params['datadir'])
    output_folder = Path(params['output_folder'])
    working_directory = Path(params['working_directory']) / 'tempDir'

    logger.info('Start loading recordings')

    recordings_list = [rec for rec in datadir.glob('*_g0')]
    # recordings_list = [datadir / '14112022_F2103_Fettucini_PM_g0']
    for rec in recordings_list:
        logger.info(f'Loading recording {rec.name}')
        try:
            spikesorting_pipeline(rec, params)

            spikesorting_postprocessing(rec, params)
        except Exception as e:
            logger.error(f'Error processing {rec.name}: {e}')

if __name__ == '__main__':
    main()