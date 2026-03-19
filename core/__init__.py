"""Core module: standalone implementations extracted from snel_toolkit."""

from .base_dataset import BaseDataset
from .nwb_dataset import NWBDataset
from .lfads_interface import LFADSInterface, SegmentRecord, chop_data, merge_chops
from .utils import parmap, rgetattr
