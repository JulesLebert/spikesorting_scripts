# Set logging before the rest as neo (and neo-based imports) needs to be imported after logging has been set
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

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.curation as scu
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp

def spikeglx_preprocessing(recording):
    # Preprocessing steps
    logger.info(f'preprocessing recording')

    # equivalent to what catgt does
    recording = spre.phase_shift(recording)
    # bandpass filter and common reference can be skipped if using kilosort as it does it internally 
    # but doesn't change anything to keep it
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = spre.common_reference(recording, reference='global', operator='median')
    return recording

def spikesorting_pipeline(rec_name, params):
    # Spikesorting pipeline for a single recording
    working_directory = Path(params['working_directory']) / 'tempDir'

    recording = se.read_spikeglx(rec_name, stream_id = 'imec0.ap')
    recording = spikeglx_preprocessing(recording)

    logger.info(f'running spike sorting')
    sorting_output = ss.run_sorters(params['sorter_list'], [recording], working_folder=working_directory,
        mode_if_folder_exists='overwrite', 
        engine='loop', verbose=True)

def spikesorting_postprocessing(params):
    jobs_kwargs = params['jobs_kwargs']
    sorting_output = ss.collect_sorting_outputs(Path(params['working_directory']))
    for (rec_name, sorter_name), sorting in sorting_output.items():
        logger.info(f'Postprocessing {rec_name} {sorter_name}')
        if params['remove_dup_spikes']:
            logger.info(f'removing duplicate spikes')
            scu.remove_duplicated_spikes(sorting, censored_period_ms=params['remove_dup_spikes']['censored_period_ms'])

        logger.info('waveform extraction')
        outDir = params['output_folder'] / rec_name / sorter_name
        we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder',
                load_if_exists=True,
                overwrite=False,
                ms_before=1, ms_after=2., max_spikes_per_unit=200,
                **jobs_kwargs)

        logger.info(f'Computing quality netrics')
        metrics = sqm.compute_quality_metrics(we, n_jobs = jobs_kwargs['n_jobs'], verbose=True)

        logger.info(f'Exporting to phy')
        sexp.export_to_phy(we, outDir / 'phy_folder', verbose=True, **jobs_kwargs)

        logger.info('Export report')
        sexp.export_report(we, outDir / 'report',
                format='png',
                force_computation=True,
                **jobs_kwargs)


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
    working_directory = Path(params['working_directory'])

    logger.info('Start loading recordings')

    # Load recordings
    sessions = [sess.name for sess in datadir.glob('*_g0')]

    recordings_list = []
    # /!\ This assumes that all the recordings must have same mapping
    for session in sessions:
        recording = se.read_spikeglx(datadir / session, stream_id='imec0.ap')
        recording = spikeglx_preprocessing(recording)
        recordings_list.append(recording)

    multirecording = sc.concatenate_recordings(recordings_list)
    multirecordings = multirecordings.set_probe(recordings_list[0].get_probe())
    sorting = ss.run_sorters(params['sorter_list'], [multirecording], working_folder=working_directory,
        mode_if_folder_exists='overwrite', 
        engine='loop', verbose=True)

    # # If recordings don't have same mapping, can do something like this:
    # # In this example, only 2 mappings are in the data, but it can be extended to more mappings
    # # To extract channel coordinates from a probe object, use probe.get_channel_locations()
    # # And then group recordings based on this
    
    # recordings_list_probemap_12 = []
    # recordings_list_probemap_34 = []

    # for session in sessions:
    #     recording = se.read_spikeglx(catgt_data / session, stream_id='imec0.ap')
    #     recording = spikeglx_preprocessing(recording)

    #     probe = recording.get_probe()
    #     if '0' in probe.shank_ids:
    #         recordings_list_probemap_12.append(recording)
    #     else:
    #         recordings_list_probemap_34.append(recording)

    # for (multirec, probemap_name) in zip(
    #     [recordings_list_probemap_12, recordings_list_probemap_34],['probemap_12', 'probemap_34']):
    # multirecordings = si.concatenate_recordings(multirec)
    # multirecordings = multirecordings.set_probe(multirec[0].get_probe())
    # multirecordings.is_filtered = True

    # sorting = si.run_kilosort3(multirecordings, output_folder=catgt_data / f'{probemap_name}_concatenated')



    # Not sure if it works with concatenated recordings
    # And might take a while to run extract waveforms
    spikesorting_postprocessing(params)

if __name__ == '__main__':
    main()