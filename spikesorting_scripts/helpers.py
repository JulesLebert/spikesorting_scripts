from pathlib import Path
import os

from .npyx_metadata_fct import load_meta_file

def get_channelmap_names(dp):
    """Get the channel map name from the meta file

    Parameters
    ----------
    dp : str
        Path to the recording folder

    Returns
    -------
    channel_map_name : dict
        
    """

    dp = Path(dp)
    imec_folders = [imec_folder for imec_folder in dp.glob('*_imec*')]
    channel_map_dict = {}

    for imec_folder in imec_folders:
        metafile = [meta for meta in next(os.walk(imec_folder))[2] if meta.endswith('.meta')]
        if len(metafile)==0:
            raise(f'No metafile found in {imec_folder.name}')
        elif len(metafile)>1:
            print(f'More that 1 metafile found in {imec_folder.name}. Using {metafile[0]}')

        meta = load_meta_file(imec_folder / metafile[0])
        channel_map_name = Path(meta['imRoFile'])
        channel_map_dict[imec_folder.name] = channel_map_name.name

    return channel_map_dict