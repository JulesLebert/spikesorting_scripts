from spikeinterface.core.core_tools import define_function_from_class

from spikeinterface.extractors.neoextractors.neobaseextractor import NeoBaseRecordingExtractor, NeoBaseSortingExtractor

from spikesorting_scripts.io.customtdtrawio import CustomTdtRawIO

class CustomTdtRecordingExtractor(NeoBaseRecordingExtractor):
    """
    Class for reading TDT folder.

    Based on :py:class:`neo.rawio.TdTRawIO`

    Parameters
    ----------
    folder_path: str
        The folder path to the tdt folder.
    stream_id: str, optional
        If there are several streams, specify the stream id you want to load.
    stream_name: str, optional
        If there are several streams, specify the stream name you want to load.
    all_annotations: bool, optional, default: False
        Load exhaustively all annotations from neo.
    """
    mode = 'folder'
    NeoRawIOClass = CustomTdtRawIO
    name = "customTdt"

    def __init__(self, folder_path, stream_id=None, stream_name=None, block_index=None, all_annotations=False,
                store=['BB_2','BB_3','BB_4','BB_5']):
        # neo_kwargs = self.map_to_neo_kwargs(folder_path)
        neo_kwargs = {'dirname': folder_path, 'store': store}
        NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, 
                                           stream_name=stream_name,
                                           block_index=block_index,
                                           all_annotations=all_annotations, 
                                           **neo_kwargs)
        self._kwargs.update(dict(folder_path=str(folder_path), stream_id=stream_id, store=store))


# class CustomTdtRecordingExtractor(NeoBaseRecordingExtractor):
#     """
#     Class for reading TDT folder
#     Based on neo.rawio.TdTRawIO
#     Parameters
#     ----------
#     folder_path: str
#         The tdt folder.
#     stream_id: str or None
#     """
#     mode = 'folder'
#     NeoRawIOClass = 'CustomTdtRawIO'

#     def __init__(self, folder_path, stream_id=None, store=['BB_2','BB_3','BB_4','BB_5']):
#         neo_kwargs = {'dirname': folder_path, 'store': store}
#         NeoBaseRecordingExtractor.__init__(self, stream_id=stream_id, **neo_kwargs)

#         self._kwargs = dict(folder_path=str(folder_path), stream_id=stream_id, store=store)

#     @classmethod
#     def map_to_neo_kwargs(cls, folder_path):
#         neo_kwargs = {'dirname': str(folder_path)}
#         return neo_kwargs

read_custom_tdt = define_function_from_class(source_class=CustomTdtRecordingExtractor, name="read_custom_tdt")
