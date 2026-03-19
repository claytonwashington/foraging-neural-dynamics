"""NWBDataset: Load and preprocess neural data from NWB files.

Adapted from snel_toolkit.datasets.nwb, which was originally based on:
- https://github.com/neurallatents/nlb_tools/blob/main/nlb_tools/nwb_interface.py
  (Original author: Felix Pei)
- https://github.com/snel-repo/snel-toolkit
  (Original author: Andrew Sedler)
"""

import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, ProcessingModule, TimeSeries
from pynwb.core import MultiContainerInterface

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class NWBDataset(BaseDataset):
    """Load and preprocess neural/behavioral data from NWB files.

    Handles single or multi-file NWB datasets, spike binning,
    timestamp alignment, and trial info extraction.
    """

    def __init__(self, fpath, prefix="", split_heldout=False, skip_fields=[], bin_width=1):
        """Initialize NWBDataset from one or more NWB files.

        Parameters
        ----------
        fpath : str
            Path to an NWB file or directory containing NWB files.
        prefix : str, optional
            Glob pattern to filter NWB files in a directory.
        split_heldout : bool, optional
            Whether to separate heldin and heldout units.
        skip_fields : list, optional
            Field names to skip during loading (saves memory).
        bin_width : int, optional
            Bin width in ms for spike binning, by default 1.
        """
        fpath = os.path.expanduser(fpath)
        self.fpath = fpath
        self.prefix = prefix
        self.bin_width = bin_width

        if not os.path.exists(fpath):
            raise FileNotFoundError("Specified file or directory not found")
        if os.path.isdir(fpath):
            filenames = sorted(glob(os.path.join(fpath, prefix + "*.nwb")))
        else:
            filenames = [fpath]

        if len(filenames) == 0:
            raise FileNotFoundError(
                f"No matching files with prefix {prefix} found in directory {fpath}"
            )
        elif len(filenames) > 1:
            loaded = [
                self.load(fname, split_heldout=split_heldout, skip_fields=skip_fields)
                for fname in filenames
            ]
            datas, trial_infos, descriptions, bin_widths = [
                list(out) for out in zip(*loaded)
            ]
            assert np.all(
                np.array(bin_widths) == bin_widths[0]
            ), "Bin widths of loaded datasets must be the same"

            def trial_shift(x, shift_ms, trial_offset):
                if x.name.endswith("_time"):
                    return x + pd.to_timedelta(shift_ms, unit="ms")
                elif x.name == "trial_id":
                    return x + trial_offset
                else:
                    return x

            past_end = datas[0].index[-1].total_seconds() + round(
                50 * bin_widths[0] / 1000, 4
            )
            descriptions_full = descriptions[0]
            tcount = len(trial_infos[0])
            for i in range(1, len(datas)):
                block_start_ms = np.ceil(past_end * 10) * 100
                datas[i] = datas[i].shift(block_start_ms, freq="ms")
                trial_infos[i] = trial_infos[i].apply(
                    trial_shift, shift_ms=block_start_ms, trial_offset=tcount
                )
                descriptions_full.update(descriptions[i])
                past_end = datas[i].index[-1].total_seconds() + round(
                    50 * bin_widths[i] / 1000, 4
                )
                tcount += len(trial_infos[i])
            self.data = pd.concat(datas, axis=0, join="outer")
            self.trial_info = pd.concat(trial_infos, axis=0, join="outer").reset_index(
                drop=True
            )
            self.descriptions = descriptions_full
            self.bin_width = bin_widths[0]
            new_index = pd.to_timedelta(
                (
                    np.arange(
                        round(
                            self.data.index[-1].total_seconds() * 1000 / self.bin_width
                        )
                        + 1
                    )
                    * self.bin_width
                ).round(4),
                unit="ms",
            )
            self.data = self.data.reindex(new_index)
            self.data.index.name = "clock_time"
        else:
            data, trial_info, descriptions, bin_width = self.load(
                filenames[0], split_heldout=split_heldout
            )
            self.data = data
            self.trial_info = trial_info
            self.descriptions = descriptions
            self.bin_width = bin_width

    def load(self, fpath, split_heldout=False, skip_fields=[]):
        """Load data from an NWB file into DataFrames.

        Parameters
        ----------
        fpath : str
            Path to the NWB file.
        split_heldout : bool, optional
            Whether to separate heldin/heldout units.
        skip_fields : list, optional
            Fields to skip during loading.

        Returns
        -------
        tuple
            (data, trial_info, descriptions, bin_width)
        """
        logger.info(f"Loading {fpath}")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        # Load trial info
        trial_info = (
            nwbfile.trials.to_dataframe()
            .reset_index()
            .rename({"id": "trial_id", "stop_time": "end_time"}, axis=1)
        )

        if nwbfile.units is not None:
            has_units = True
            units = nwbfile.units.to_dataframe()
            unit_info = nwbfile.units.electrodes.to_dataframe()
            unit_info = unit_info.drop(
                columns=unit_info.columns.difference(["group_name", "location"])
            )
            if any(units.columns.isin(["location"])):
                unit_info = unit_info.reset_index().rename(columns={"id": "elec_id"})
                unit_info = unit_info.drop(columns="location")
                unit_info = pd.concat([unit_info, units.location], axis=1)
                unit_info.index.name = "unit_id"
            self.unit_info = unit_info
        else:
            has_units = False
            acq_key = list(nwbfile.acquisition.keys())[0]
            name_key = list(nwbfile.acquisition[acq_key].time_series.keys())[0]
            dt_s = np.diff(nwbfile.acquisition[acq_key][name_key].timestamps).round(4)
            bin_width = np.unique(dt_s)[0] * 1000

        # Load descriptions
        descriptions = {}
        for name, info in zip(nwbfile.trials.colnames, nwbfile.trials.columns):
            descriptions[name] = info.description

        # Find all timeseries
        def make_df(ts):
            """Convert a TimeSeries to a pandas DataFrame."""
            if ts.timestamps is not None:
                index = ts.timestamps[()]
            else:
                index = np.arange(ts.data.shape[0]) / ts.rate + ts.starting_time
            columns = (
                ts.comments.split("[")[-1].split("]")[0].split(",")
                if "columns=" in ts.comments
                else None
            )
            if len(ts.data.shape) > 1:
                base_column_name = columns[0]
                columns = []
                for i in range(ts.data.shape[1]):
                    columns.append(base_column_name + "_" + str(i))
            df = pd.DataFrame(
                ts.data[()], index=pd.to_timedelta(index, unit="s"), columns=columns
            )
            return df

        def find_timeseries(nwbobj):
            """Recursively search NWB file for time series data."""
            ts_dict = {}
            for child in nwbobj.children:
                if isinstance(child, TimeSeries):
                    if child.name in skip_fields:
                        continue
                    ts_dict[child.name] = make_df(child)
                    descriptions[child.name] = child.description
                elif isinstance(child, ProcessingModule):
                    pm_dict = find_timeseries(child)
                    ts_dict.update(pm_dict)
                elif isinstance(child, MultiContainerInterface):
                    name = child.name
                    ts_dfs = []
                    for field in child.children:
                        if isinstance(field, TimeSeries):
                            if name in skip_fields:
                                continue
                            ts_dfs.append(make_df(field))
                    ts_dict[name] = pd.concat(ts_dfs, axis=1)
                    descriptions[name] = field.description
            return ts_dict

        data_dict = find_timeseries(nwbfile)

        if has_units:
            # Calculate timestamps
            start_time = 0.0
            bin_width = self.bin_width
            rate = round(1000.0 / bin_width, 2)
            end_time = (
                round(max(units.obs_intervals.apply(lambda x: x[-1][-1])) * rate)
                * bin_width
            )
            if end_time < trial_info["end_time"].iloc[-1]:
                end_time = round(trial_info["end_time"].iloc[-1] * rate) * bin_width
            timestamps = (np.arange(start_time, end_time, bin_width) / 1000).round(6)
            timestamps_td = pd.to_timedelta(timestamps, unit="s")

            # Validate timeseries timestamps
            for key, val in list(data_dict.items()):
                if not np.all(
                    np.isin(np.round(val.index.total_seconds(), 6), timestamps)
                ):
                    logger.warning(f"Dropping {key} due to timestamp mismatch.")
                    data_dict.pop(key)

            def make_mask(obs_intervals):
                """Create bool mask for spikes outside observation intervals."""
                mask = np.full(timestamps.shape, True)
                for start, end in obs_intervals:
                    start_idx = np.ceil(
                        round((start - timestamps[0]) * rate, 6)
                    ).astype(int)
                    end_idx = np.floor(round((end - timestamps[0]) * rate, 6)).astype(
                        int
                    )
                    mask[start_idx:end_idx] = False
                return mask

            # Bin spikes
            masks = (
                [(~units.heldout).to_numpy(), units.heldout.to_numpy()]
                if split_heldout
                else [np.full(len(units), True)]
            )

            for mask, name in zip(masks, ["spikes", "heldout_spikes"]):
                if not np.any(mask):
                    continue
                spike_arr = np.full(
                    (len(timestamps), np.sum(mask)), 0.0, dtype="float16"
                )
                for idx, (_, unit) in enumerate(units[mask].iterrows()):
                    spike_idx, spike_cnt = np.unique(
                        ((unit.spike_times - timestamps[0]) * rate)
                        .round(6)
                        .astype(int),
                        return_counts=True,
                    )
                    spike_arr[spike_idx, idx] = spike_cnt
                if "obs_intervals" in units.columns:
                    neur_mask = make_mask(units[mask].iloc[0].obs_intervals)
                    if np.any(spike_arr[neur_mask]):
                        logger.warning("Spikes found outside of observed interval.")
                    spike_arr[neur_mask] = np.nan
                data_dict[name] = pd.DataFrame(
                    spike_arr, index=timestamps_td, columns=units[mask].index
                ).astype("float64", copy=False)

        # Create MultiIndex columns
        data_list = []
        for key, val in data_dict.items():
            chan_names = None if type(val.columns) == pd.RangeIndex else val.columns
            val.columns = self._make_midx(
                key, chan_names=chan_names, num_channels=val.shape[1]
            )
            data_list.append(val)

        data = pd.concat(data_list, axis=1)
        data.index.name = "clock_time"
        data.sort_index(axis=1, inplace=True)

        # Convert time fields to timedelta
        def to_td(x):
            if x.name.endswith("_time"):
                return pd.to_timedelta(x, unit="s")
            else:
                return x

        trial_info = trial_info.apply(to_td, axis=0)
        io.close()

        return data, trial_info, descriptions, bin_width
