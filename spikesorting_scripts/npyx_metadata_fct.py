# Functions adapted from NeuroPyxels package (https://github.com/m-beau/NeuroPyxels)
# To read metadata from neuropixels recordings

from pathlib import Path
import os
import numpy as np
from ast import literal_eval as ale


def get_npix_sync(dp, output_binary=False, unit='seconds', sync_trial_chan=[5], verbose=True):
    '''Added by Jules
    Get sync pulses aligned on trial onset in nidq.bin file (channel 6 in TDT = 5 in python)
    Save a .txt as catgt would do
    Parameters:
        - dp: str, datapath
        - output_binary: bool, whether to output binary sync channel as 0/1s    

    Returns:
        Dictionnaries of length n_channels = number of channels where threshold crossings were found, [0-16]
        - onsets: dict, {channel_i:np.array(onset1, onset2, ...), ...} in 'unit'
        - offsets: dict, {channel_i:np.array(offset1, offset2, ...), ...} in 'unit'

    '''

    dp=Path(dp)

    onsets={}
    offsets={}
    fname=''
    # sync_dp=dp/'sync_chan'


    nidqpath = dp
    metafile = [meta for meta in next(os.walk(nidqpath))[2] if meta.endswith('.meta')]
    if len(metafile)==0:
        raise('No metafile found in {}'.format(nidqpath))
    elif len(metafile)>1:
        if verbose: print('More that 1 metafile found in {}. Using {}'.format(nidqpath,metafile[0]))
    metafile = nidqpath / metafile[0]

    # load metafile
    meta={}
    with open(metafile, 'r') as f:
        for ln in f.readlines():
            tmp = ln.split('=')
            k, val = tmp[0], ''.join(tmp[1:])
            k = k.strip()
            val = val.strip('\r\n')
            if '~' in k:
                meta[k] = val.strip('(').strip(')').split(')(')
            else:
                try:  # is it numeric?
                    meta[k] = float(val)
                except:
                    meta[k] = val


    srate = meta['niSampRate'] if unit=='seconds' else 1

    if fname=='':
        fname = f'{dp.name}_t0.nidq.bin'

        nchan = meta['nSavedChans']
        dt = np.dtype('int16')

        nsamples=os.path.getsize(nidqpath/fname)/ (nchan * dt.itemsize)
        syncdat=np.memmap(nidqpath/fname,
                    mode='r',
                    dtype=dt,
                    shape=(int(nsamples), int(nchan)))[:,-1]
        
        print('Unpacking {}...'.format(fname))
        binary = unpackbits(syncdat.flatten(),16).astype(np.int8)
        # sync_fname = fname[:-4]+'_sync'
        # np.savez_compressed(sync_dp/(sync_fname+'.npz'), binary)   

    if output_binary:
        return binary

    # Generates onsets and offsets from binary
    mult = 1
    sync_idx_onset = np.where(mult*np.diff(binary, axis = 0)>0)
    sync_idx_offset = np.where(mult*np.diff(binary, axis = 0)<0)
    # Only loads the sync channel (6 in Nellie)
    for ichan in sync_trial_chan: #np.unique(sync_idx_onset[1]):
        ons = sync_idx_onset[0][
            sync_idx_onset[1] == ichan]
        onsets[ichan] = ons
        # np.save(Path(sync_dp, sync_fname+'{}on_samples.npy'.format(ichan)), ons)
    for ichan in sync_trial_chan: #np.unique(sync_idx_offset[1]):
        ofs = sync_idx_offset[0][
            sync_idx_offset[1] == ichan]
        offsets[ichan] = ofs
        # np.save(Path(sync_dp, sync_fname+'{}of_samples.npy'.format(ichan)), ofs)

    onsets={ok:ov/srate for ok, ov in onsets.items()}
    offsets={ok:ov/srate for ok, ov in offsets.items()}

    assert any(onsets), ("WARNING no sync channel found in dataset - "
        "make sure you are running this function on a dataset with a synchronization TTL!")

    for ichan in sync_trial_chan:
        np.savetxt(dp/f'{dp.name}_tcat.nidq.XD_4_{ichan}_100.txt', onsets[ichan], fmt='%1.4f')

    return onsets,offsets


def list_files(directory, extension, full_path=False):
    directory=str(directory)
    files = [f for f in os.listdir(directory) if f.endswith('.' + extension)]
    files.sort()
    if full_path:
        return [Path('/'.join([directory,f])) for f in files]
    return files

def unpackbits(x,num_bits = 16):
    '''
    unpacks numbers in bits.
    '''
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(np.int64).reshape(xshape + [num_bits])