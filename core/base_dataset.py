"""Base dataset class for neural data analysis.

Extracted from snel_toolkit.datasets.base. Contains BaseDataset with
methods for data initialization, resampling, smoothing, cross-correlation
rejection, and continuous data management.

Original authors: SNEL group (Emory University)
"""

import copy
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.signal as signal

from .utils import parmap

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for neural datasets.

    Provides standard functionality for data management including
    resampling, smoothing, cross-correlation analysis, and data
    merging. Subclass this for specific data formats (e.g., NWB).
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def load(self):
        """Load data and initialize self.data, self.trial_info, and self.bin_width."""
        pass

    @property
    def bin_size(self):
        """Calculate the bin size in milliseconds from the data index."""
        return (self.data.index[1] - self.data.index[0]).total_seconds() * 1000

    def init_data_from_dict(
        self,
        data_dict,
        bin_width,
        name_dict={},
        trial_info=pd.DataFrame(),
        time_stamps=None,
    ):
        """Initialize continuous DataFrame from a dictionary of arrays.

        Parameters
        ----------
        data_dict : dict
            Maps signal_type names to numpy arrays (time-major, same row count).
        bin_width : float
            Sampling interval in seconds.
        name_dict : dict, optional
            Maps signal_type names to lists of column names.
        trial_info : pd.DataFrame, optional
            Trial metadata.
        time_stamps : np.ndarray, optional
            Time steps in seconds. Must match data length.
        """
        self.bin_width = bin_width
        self.trial_info = trial_info
        for signal_type, data in data_dict.items():
            if signal_type not in name_dict:
                chan_nums = range(data.shape[1])
                name_dict[signal_type] = [f"{x:04d}" for x in chan_nums]
        frames = []
        for signal_type, channels in name_dict.items():
            midx = pd.MultiIndex.from_product(
                [[signal_type], channels], names=("signal_type", "channel")
            )
            sig = data_dict[signal_type]
            if time_stamps is None:
                time_stamps = bin_width * np.arange(len(sig))
            signal_type_data = pd.DataFrame(sig, index=time_stamps, columns=midx)
            signal_type_data.index.name = "clock_time"
            frames.append(signal_type_data.copy())
        self.data = pd.concat(frames, axis=1)
        self.data.index = pd.to_timedelta(self.data.index, unit="s")
        n_rows, n_cols = self.data.shape
        logger.info(f"Initialized `self.data` with {n_rows} rows and {n_cols} columns.")

    def resample(self, target_bin):
        """Rebin spikes and downsample continuous signals.

        Parameters
        ----------
        target_bin : float
            Target bin size in ms. Must be an integer multiple of self.bin_width.
        """
        logger.info(f"Resampling data to {target_bin} ms.")
        if target_bin == self.bin_width:
            logger.warning(f"Dataset already at {target_bin} ms resolution, skipping.")
            return
        resample_factor = target_bin / self.bin_width
        assert resample_factor.is_integer(), (
            "target_bin must be an integer multiple of bin_width."
        )

        def resample_column(x):
            resamp = x.resample("{}S".format(target_bin / 1000)).sum()
            resamp = resamp[: int(np.ceil(len(x) / resample_factor))]
            signal_type = x.name[0]
            if "spikes" not in signal_type:
                decimated_x = signal.decimate(
                    x, int(resample_factor), n=500, ftype="fir"
                )
                resamp = pd.Series(decimated_x, index=resamp.index)
            return resamp

        self.data = self.data.apply(resample_column)
        self.bin_width = target_bin

    def smooth_spk(self, gauss_width, signal_type="spikes", name=None, overwrite=False):
        """Apply Gaussian smoothing to spike data (or any signal type).

        Parameters
        ----------
        gauss_width : int
            Standard deviation of the Gaussian kernel in ms.
        signal_type : str, optional
            Signal group to smooth, by default 'spikes'.
        name : str, optional
            Suffix for the new smoothed signal name.
        overwrite : bool, optional
            If True, overwrite the original signal.
        """
        assert name or overwrite, (
            "You must either provide a name for the smoothed "
            "data or specify to overwrite the existing data."
        )
        logger.info(f"Smoothing {signal_type} with a {gauss_width} ms Gaussian.")
        gauss_bin_std = gauss_width / self.bin_width
        win_len = int(6 * gauss_bin_std)
        window = signal.gaussian(win_len, gauss_bin_std, sym=True)
        window /= np.sum(window)
        spike_vals = self.data[signal_type].values
        spike_vals = [spike_vals[:, i] for i in range(spike_vals.shape[1])]

        def filt(args):
            x, window = args
            y = signal.lfilter(window, 1.0, x)
            shift_len = len(window) // 2
            y = np.concatenate([y[shift_len:], np.full(shift_len, np.nan)], axis=0)
            return y

        y_list = parmap(filt, zip(spike_vals, [window for _ in range(len(spike_vals))]))
        col_names = self.data[signal_type].columns
        if name is None and overwrite:
            smoothed_name = signal_type
        elif name:
            smoothed_name = signal_type + "_" + name
        else:
            assert 0, "Either name or overwrite should be set!"

        for col, v in zip(col_names, y_list):
            self.data[smoothed_name, col] = v

    def get_pair_xcorr(
        self,
        signal_type,
        threshold=None,
        zero_chans=False,
        channels=None,
        max_points=None,
        removal="corr",
    ):
        """Calculate pairwise cross-correlations and optionally remove correlated channels.

        Parameters
        ----------
        signal_type : str
            Signal type to analyze (usually 'spikes').
        threshold : float, optional
            Correlation threshold above which to remove channels.
        zero_chans : bool, optional
            If True, zero out channels instead of removing them.
        channels : list, optional
            Not implemented. Specific channels to analyze.
        max_points : int, optional
            Number of initial data points to use.
        removal : {'corr', 'rate'}
            Strategy for removing correlated channels.

        Returns
        -------
        tuple
            (pair_corr, chan_names_to_drop): list of ((i,k), corr) pairs and dropped names.
        """
        assert removal in ["corr", "rate"]
        if max_points is not None:
            data = self.data[:max_points]
        else:
            data = self.data
        if channels is not None:
            raise NotImplementedError

        np_data = data[signal_type].values
        chan_names = data[signal_type].columns
        n_dim = np_data.shape[1]
        pairs = [(i, k) for i in range(n_dim) for k in range(i)]

        def xcorr_func(args):
            i, k = args
            c = np.sum(np_data[:, i] * np_data[:, k]).astype(np.float32)
            if c == 0:
                return 0.0
            c /= np.sqrt(np.sum(np_data[:, i] ** 2) * np.sum(np_data[:, k] ** 2))
            return c

        corr_list = parmap(xcorr_func, pairs)
        pair_corr = list(zip(pairs, corr_list))

        chan_names_to_drop = []
        if threshold:
            pair_corr_tmp = copy.deepcopy(pair_corr)
            if removal == "corr":
                pair_corr_tmp.sort(key=lambda x: x[1], reverse=False)
                while pair_corr_tmp:
                    pair, corr = pair_corr_tmp.pop(-1)
                    if corr > threshold:
                        c1 = [p[1] for p in pair_corr_tmp if pair[0] in p[0]]
                        c2 = [p[1] for p in pair_corr_tmp if pair[1] in p[0]]
                        cnt1 = sum(1 for c in c1 if c > threshold)
                        cnt2 = sum(1 for c in c2 if c > threshold)
                        if cnt1 > cnt2:
                            chan_dropp = pair[0]
                        elif cnt1 < cnt2:
                            chan_dropp = pair[1]
                        else:
                            if np.mean(c1) > np.mean(c2):
                                chan_dropp = pair[0]
                            else:
                                chan_dropp = pair[1]
                        pair_corr_tmp = [
                            p for p in pair_corr_tmp if chan_dropp not in p[0]
                        ]
                        chan_names_to_drop.append(chan_names[chan_dropp])
            elif removal == "rate":
                neuron_rates = np.mean(np_data, axis=0)
                high_corr_pairs = [
                    pair for pair, corr in pair_corr_tmp if corr > threshold
                ]
                while high_corr_pairs:
                    high_corr_channels = np.unique(np.concatenate(high_corr_pairs))
                    high_corr_rates = neuron_rates[high_corr_channels]
                    drop_neuron = high_corr_channels[np.argmax(high_corr_rates)]
                    high_corr_pairs = [
                        p for p in high_corr_pairs if drop_neuron not in p
                    ]
                    chan_names_to_drop.append(chan_names[drop_neuron])

            if zero_chans:
                logger.info(f"Zeroing channel names: {chan_names_to_drop}")
                for col in chan_names_to_drop:
                    self.data[signal_type, col] = 0
            else:
                logger.info(f"Removing channel names: {chan_names_to_drop}")
                self.data.drop(
                    [(signal_type, cc) for cc in chan_names_to_drop],
                    axis=1,
                    inplace=True,
                )
                self.data.columns = self.data.columns.remove_unused_levels()

        return pair_corr, chan_names_to_drop

    def _make_midx(self, signal_type, chan_names=None, num_channels=None):
        """Create a pd.MultiIndex for a given signal_type.

        Parameters
        ----------
        signal_type : str
            Name of the signal group.
        chan_names : list, optional
            Custom channel names.
        num_channels : int, optional
            Number of channels (used when chan_names is None).

        Returns
        -------
        pd.MultiIndex
            MultiIndex with (signal_type, channel) levels.
        """
        if chan_names is not None:
            assert len(chan_names) == num_channels
        elif "rates" in signal_type:
            chan_names = self.data.spikes.columns
        else:
            chan_names = [f"{i:04d}" for i in range(num_channels)]
        midx = pd.MultiIndex.from_product(
            [[signal_type], chan_names], names=("signal_type", "channel")
        )
        return midx

    def add_continuous_data(self, cts_data, signal_type, chan_names=None):
        """Add a continuous data field to the main DataFrame.

        Parameters
        ----------
        cts_data : np.ndarray
            Array whose first dimension matches self.data row count.
        signal_type : str
            Label for this group of signals.
        chan_names : list, optional
            Channel names for this data.
        """
        logger.info(f"Adding continuous {signal_type} to the main DataFrame.")
        midx = self._make_midx(signal_type, chan_names, cts_data.shape[1])
        new_data = pd.DataFrame(cts_data, index=self.data.index, columns=midx)
        self.data = pd.concat([self.data, new_data], axis=1)
