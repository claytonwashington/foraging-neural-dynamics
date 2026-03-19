"""LFADS Interface: chop continuous data for LFADS and merge outputs back.

Extracted from snel_toolkit.interfaces. Provides LFADSInterface for
breaking continuous neural data into overlapping segments (chops) for
LFADS modeling, and reassembling processed outputs back into the
original continuous/trialized format.

NOTE: The LFADSInterface object used for chopping must be preserved
(e.g., pickled) and reused when merging LFADS outputs.
"""

import logging
import os
from collections import defaultdict
from os import path

import h5py
import numpy as np
import pandas as pd

# Standard data names expected by LFADS
DATA_NAME = "data"
EXT_INPUT_NAME = "ext_input"
INDEX_NAME = "inds"

# Default mapping from signal_types to HDF5 data labels
DEFAULT_CHOP_MAP = {"spikes": DATA_NAME}

# Default mapping from LFADS output fields to signal_type names
DEFAULT_MERGE_MAP = {
    "rates": "lfads_rates",
    "factors": "lfads_factors",
    "gen_inputs": "lfads_gen_inputs",
}

logger = logging.getLogger(__name__)


class SegmentRecord:
    """Stores information needed to reconstruct a segment from chops.

    Each segment corresponds to one trial (trialized data) or the entire
    continuous recording (continuous data).
    """

    def __init__(self, seg_id, clock_time, offset, n_chops, overlap):
        """Initialize a SegmentRecord.

        Parameters
        ----------
        seg_id : int
            ID of this segment (trial_id or 1 for continuous).
        clock_time : pd.Series
            TimeDeltaIndex of the original data for this segment.
        offset : int
            Offset of the first chop from segment start (in bins).
        n_chops : int
            Number of chops in this segment.
        overlap : int
            Overlap between adjacent chops (in bins).
        """
        self.seg_id = seg_id
        self.clock_time = clock_time
        self.offset = offset
        self.n_chops = n_chops
        self.overlap = overlap

    def rebuild_segment(self, chops, smooth_pwr=2):
        """Reassemble a segment from its chops.

        Parameters
        ----------
        chops : np.ndarray
            3D array (n_chops × seg_len × data_dim) of chopped data.
        smooth_pwr : float, optional
            Power for overlap smoothing (see merge_chops), by default 2.

        Returns
        -------
        pd.DataFrame
            Reconstructed segment indexed by original clock_time.
        """
        merged_array = merge_chops(
            chops,
            overlap=self.overlap,
            orig_len=len(self.clock_time) - self.offset,
            smooth_pwr=smooth_pwr,
        )
        # Pad with NaNs for unmodeled offset region
        data_dim = merged_array.shape[1]
        offset_nans = np.full((self.offset, data_dim), np.nan)
        merged_array = np.concatenate([offset_nans, merged_array])
        segment_df = pd.DataFrame(merged_array, index=self.clock_time)
        return segment_df


class LFADSInterface:
    """Interface for chopping neural data for LFADS and merging outputs.

    Handles the complete workflow of:
    1. Breaking continuous data into overlapping time windows (chops)
    2. Saving chops as HDF5 files for lfads-torch
    3. Merging LFADS outputs back into the original data format
    """

    def __init__(
        self,
        window,
        overlap,
        max_offset=0,
        chop_margins=0,
        random_seed=None,
        chop_fields_map=DEFAULT_CHOP_MAP,
        merge_fields_map=DEFAULT_MERGE_MAP,
    ):
        """Initialize LFADSInterface.

        Parameters
        ----------
        window : int
            Length of chopped segments in ms.
        overlap : int
            Overlap between chopped segments in ms.
        max_offset : int, optional
            Maximum random offset of first chop from segment start (ms).
        chop_margins : int, optional
            Extra margin bins on each end of chops (for temporal_shift in LFADS).
        random_seed : int, optional
            Random seed for reproducibility.
        chop_fields_map : dict, optional
            Maps DataFrame column groups to LFADS input field names.
        merge_fields_map : dict, optional
            Maps LFADS output fields to DataFrame signal_type names.
        """

        def to_timedelta(ms):
            return pd.to_timedelta(ms, unit="ms")

        self.window = to_timedelta(window)
        self.overlap = to_timedelta(overlap)
        self.max_offset = to_timedelta(max_offset)
        self.chop_margins = chop_margins
        self.random_seed = random_seed
        self.chop_fields_map = chop_fields_map
        self.merge_fields_map = merge_fields_map

    def chop(self, neural_df):
        """Chop a continuous or trialized DataFrame into overlapping segments.

        Parameters
        ----------
        neural_df : pd.DataFrame
            Continuous or trialized neural data DataFrame.

        Returns
        -------
        dict of np.ndarray
            Data dict with LFADS field names mapping to 3D arrays
            (samples × time × features).
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        fields_map = self.chop_fields_map
        data_fields = sorted(fields_map.keys())

        def get_field_dim(field):
            return len(getattr(neural_df, field).columns)

        data_dims = [get_field_dim(f) for f in data_fields]
        data_splits = data_dims[:-1]
        input_fields = [fields_map[f] for f in data_fields]

        logger.info(
            f"Mapping data field(s) {data_fields} to LFADS input "
            f"field(s) {input_fields} with dimension(s) {data_dims}."
        )

        # Set up segments for chopping
        if "trial_id" in neural_df:
            bin_width = neural_df.clock_time.iloc[1] - neural_df.clock_time.iloc[0]
            segments = neural_df.groupby("trial_id")
        else:
            bin_width = neural_df.index[1] - neural_df.index[0]
            segments = {1: neural_df.reset_index()}.items()

        window = int(self.window / bin_width)
        overlap = int(self.overlap / bin_width)
        chop_margins_td = pd.to_timedelta(self.chop_margins * bin_width, unit="ms")

        if "trial_id" in neural_df:
            max_offset = int(self.max_offset / bin_width)
            max_offset_td = self.max_offset

            def get_offset():
                return np.random.randint(max_offset + 1)
        else:
            max_offset = 0
            max_offset_td = pd.to_timedelta(max_offset)

            def get_offset():
                return 0

            if self.max_offset > pd.to_timedelta(0):
                logger.info("Ignoring offset for continuous data.")

        def to_ms(timedelta):
            return int(timedelta.total_seconds() * 1000)

        chop_message = " - ".join([
            "Chopping data for LFADS",
            f"Window: {window} bins, {to_ms(self.window)} ms",
            f"Overlap: {overlap} bins, {to_ms(self.overlap)} ms",
            f"Max offset: {max_offset} bins, {to_ms(max_offset_td)} ms",
            f"Chop margins: {self.chop_margins} bins, {to_ms(chop_margins_td)} ms",
        ])
        logger.info(chop_message)

        data_dict = defaultdict(list)
        segment_records = []
        for segment_id, segment_df in segments:
            data_arrays = [getattr(segment_df, f).values for f in data_fields]
            segment_array = np.concatenate(data_arrays, axis=1)
            if self.chop_margins > 0:
                seg_dim = segment_array.shape[1]
                pad = np.full((self.chop_margins, seg_dim), 0.0001)
                segment_array = np.concatenate([pad, segment_array, pad])
            offset = get_offset()
            chops = chop_data(
                segment_array,
                overlap + 2 * self.chop_margins,
                window + 2 * self.chop_margins,
                offset,
            )
            data_chops = np.split(chops, np.cumsum(data_splits), axis=2)
            for field, data_chop in zip(input_fields, data_chops):
                data_dict[field].append(data_chop)
            seg_rec = SegmentRecord(
                segment_id, segment_df.clock_time, offset, len(chops), overlap
            )
            segment_records.append(seg_rec)

        self.segment_records = segment_records
        data_dict = {name: np.concatenate(c) for name, c in data_dict.items()}

        dict_key = list(data_dict.keys())[0]
        n_chops = len(data_dict[dict_key])
        n_segments = len(segment_records)
        logger.info(f"Created {n_chops} chops from {n_segments} segment(s).")

        return data_dict

    def chop_and_save(
        self,
        neural_df,
        fname,
        valid_ratio=0.2,
        valid_block=1,
        heldin_ratio=1.0,
        overwrite=False,
    ):
        """Chop data and save as HDF5 file for lfads-torch.

        Parameters
        ----------
        neural_df : pd.DataFrame
            Continuous or trialized neural data.
        fname : str
            Output HDF5 file path.
        valid_ratio : float, optional
            Fraction of data for validation, by default 0.2.
        valid_block : int, optional
            Block size for validation splitting, by default 1.
        heldin_ratio : float, optional
            Fraction for heldin subset (< 1.0 creates a separate file).
        overwrite : bool, optional
            Whether to overwrite existing files.
        """
        if not overwrite and path.isfile(fname):
            raise AssertionError(
                f"File {fname} already exists. Set `overwrite`=True to overwrite."
            )
        data_dir = path.dirname(fname)
        os.makedirs(data_dir, exist_ok=True)

        data_dict = self.chop(neural_df)
        dict_key = list(data_dict.keys())[0]
        n_samples = len(data_dict[dict_key])

        # Blocked train/valid split
        n_blocks = np.ceil(n_samples / valid_block).astype(int)
        block_nums = np.repeat(np.arange(n_blocks), valid_block)
        in_valid = block_nums % np.round(1 / valid_ratio) == 0
        in_valid = in_valid[:n_samples]
        (valid_inds,) = np.where(in_valid)
        (train_inds,) = np.where(~in_valid)

        def save_data(fname, train_inds, valid_inds):
            with h5py.File(fname, "w") as h5file:
                for ind_name, inds in zip(
                    ["train_", "valid_"], [train_inds, valid_inds]
                ):
                    h5file.create_dataset(ind_name + INDEX_NAME, data=inds)
                    for data_tag, samples in data_dict.items():
                        h5file.create_dataset(ind_name + data_tag, data=samples[inds])
            logger.info(
                f"Successfully wrote {len(train_inds)} train and "
                f"{len(valid_inds)} valid samples to {fname}."
            )

        save_data(fname, train_inds, valid_inds)

        if heldin_ratio < 1.0:
            dirname, basename = path.dirname(fname), path.basename(fname)
            heldin_fname = path.join(dirname, "heldin_" + basename)
            n_train, n_valid = len(train_inds), len(valid_inds)
            in_heldin_train = np.arange(n_train) % np.round(1 / heldin_ratio) == 0
            in_heldin_valid = np.arange(n_valid) % np.round(1 / heldin_ratio) == 0
            heldin_train_inds = train_inds[np.where(in_heldin_train)[0]]
            heldin_valid_inds = valid_inds[np.where(in_heldin_valid)[0]]
            save_data(heldin_fname, heldin_train_inds, heldin_valid_inds)

    def merge(self, data_dict, smooth_pwr=2):
        """Merge chopped LFADS outputs back into continuous format.

        Parameters
        ----------
        data_dict : dict of np.ndarray
            LFADS output arrays keyed by field name (samples × time × features).
        smooth_pwr : float, optional
            Smoothing power for overlap blending (see merge_chops).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame indexed by original clock_time with MultiIndex columns.
        """
        fields_map = self.merge_fields_map
        output_fields = sorted(fields_map.keys())
        output_arrays = [data_dict[f] for f in output_fields]
        output_dims = [a.shape[-1] for a in output_arrays]
        output_full = np.concatenate(output_arrays, axis=-1)

        seg_splits = np.cumsum([s.n_chops for s in self.segment_records])[:-1]
        seg_chops = np.split(output_full, seg_splits, axis=0)

        segment_dfs = [
            record.rebuild_segment(chops, smooth_pwr)
            for record, chops in zip(self.segment_records, seg_chops)
        ]
        merged_df = pd.concat(segment_dfs)

        signal_types = [fields_map[f] for f in output_fields]
        midx_tuples = [
            (sig, f"{i:04}")
            for sig, dim in zip(signal_types, output_dims)
            for i in range(dim)
        ]
        merged_df.columns = pd.MultiIndex.from_tuples(midx_tuples)

        return merged_df


# ==================== STATELESS FUNCTIONS ====================


def chop_data(data, overlap, window, offset=0):
    """Break continuous data into overlapping segments using stride tricks.

    Parameters
    ----------
    data : np.ndarray
        T×N array of N features across T time points.
    overlap : int
        Number of overlapping points between segments.
    window : int
        Number of time points per segment.
    offset : int, optional
        Starting offset (breaks temporal connection), by default 0.

    Returns
    -------
    np.ndarray
        S×T×N array of S overlapping segments.
    """
    offset_data = data[offset:]
    shape = (
        int((offset_data.shape[0] - offset - overlap) / (window - overlap)),
        window,
        offset_data.shape[-1],
    )
    strides = (
        offset_data.strides[0] * (window - overlap),
        offset_data.strides[0],
        offset_data.strides[1],
    )
    chopped = (
        np.lib.stride_tricks.as_strided(offset_data, shape=shape, strides=strides)
        .copy()
        .astype("f")
    )
    return chopped


def merge_chops(data, overlap, orig_len=None, smooth_pwr=2):
    """Merge overlapping segments back into continuous data.

    Uses a power-function ramp to smoothly blend overlapping regions.

    Parameters
    ----------
    data : np.ndarray
        S×T×N array of S overlapping segments.
    overlap : int
        Number of overlapping points between segments.
    orig_len : int, optional
        Original continuous data length (pads with NaN if needed).
    smooth_pwr : float, optional
        Power for the blending ramp. 1=linear, 2=slight preference for
        segment ends, np.inf=keep only ends, by default 2.

    Returns
    -------
    np.ndarray
        T×N array of merged continuous data.
    """
    if smooth_pwr < 1:
        logger.warning("Using `smooth_pwr` < 1 for merging chops is not recommended.")

    merged = []
    full_weight_len = data.shape[1] - 2 * overlap
    if overlap > 0:
        x = np.linspace(1 / overlap, 1 - 1 / overlap, overlap)
        ramp = 1 - x ** smooth_pwr
    else:
        ramp = np.full(0, np.nan)
    ramp = np.expand_dims(ramp, axis=-1)

    split_ixs = np.cumsum([overlap, full_weight_len])
    for i in range(len(data)):
        first, middle, last = np.split(data[i], split_ixs)
        if i == 0:
            last = last * ramp
        elif i == len(data) - 1:
            first = first * (1 - ramp) + merged.pop(-1)
        else:
            first = first * (1 - ramp) + merged.pop(-1)
            last = last * ramp
        merged.extend([first, middle, last])

    if len(merged) < 1:
        n_samples, _, data_dim = data.shape
        merged = [np.empty((n_samples, data_dim))]
    merged = np.concatenate(merged)

    if orig_len is not None and len(merged) < orig_len:
        nans = np.full((orig_len - len(merged), merged.shape[1]), np.nan)
        merged = np.concatenate([merged, nans])

    return merged
