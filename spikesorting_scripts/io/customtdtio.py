from neo.io.basefromrawio import BaseFromRaw
from spikesorting_scripts.io.customtdtrawio import CustomTdtRawIO


class CustomTdtIO(CustomTdtRawIO, BaseFromRaw):
    name = 'Tdt IO'
    description = "Tdt IO"
    mode = 'dir'

    def __init__(self, dirname,store=None):
        CustomTdtRawIO.__init__(self, dirname=dirname, store=store)
        BaseFromRaw.__init__(self, dirname)