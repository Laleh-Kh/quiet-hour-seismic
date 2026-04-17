 """
    Quiet-hour selection using the same two-stage logic as the notebook:
    1. Compute STD_SUM = STD_Z + STD_N + STD_E for each candidate window
    2. Select the quiet window either by:
       - direct minimum of STD_SUM ("first_metric"), or
       - minimum of the rolling sum of STD_SUM ("secondary_metric")

    Next-day behavior:
    - if next-day data exist, append the first append_next_day_seconds
    - if they do not exist, warn and continue with the current day only
    """

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client
from scipy.signal import spectrogram

mpl.rcParams["agg.path.chunksize"] = 10000
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


# ============================================================
# CONFIG
# ============================================================

@dataclass
class QuietHourConfig:

    # Data source: "local", "fdsn", or "stream"
    data_source: str = "stream"

    # Preloaded ObsPy streams for stream mode
    input_stream: Optional[Stream] = None
    input_stream_next_day: Optional[Stream] = None

    # Optional built-in public example streams for stream mode
    example_stream_urls_current: Optional[Tuple[str, str, str]] = (
        "https://examples.obspy.org/COP.BHE.DK.2009.050",
        "https://examples.obspy.org/COP.BHN.DK.2009.050",
        "https://examples.obspy.org/COP.BHZ.DK.2009.050",
    )
    example_stream_urls_next_day: Optional[Tuple[str, str, str]] = (
        "https://examples.obspy.org/COP.BHE.DK.2009.051",
        "https://examples.obspy.org/COP.BHN.DK.2009.051",
        "https://examples.obspy.org/COP.BHZ.DK.2009.051",
    )

    # Local input
    base_data_dir: str = ""
    month: str = ""
    file_pattern: str = "{station}.{component}.{date}.sac"

    # Public/FDSN input
    fdsn_client: str = "IRIS"
    network: str = "IU"
    station: str = "ANMO"
    location: Optional[str] = None
    channels: Optional[Tuple[str, str, str]] = None
    channel_band_priority: Tuple[str, ...] = ("BH", "HH", "EH", "HN", "EN")

    # Date
    date_str: str = "2009-02-19"  # YYYY-MM-DD

    # Quiet-hour selection
    window_length_sec: int = 3600
    step_sec: int = 60
    append_next_day_seconds: int = 3600
    selection_mode: str = "secondary_metric"  # "first_metric" or "secondary_metric"

    # Output
    output_dir: str = "quiet_hour_output"

    # Optional cache for downloaded public data
    save_downloaded_waveforms: bool = False
    downloaded_format: str = "MSEED"

    # Plot style
    figure_dpi: int = 600
    title_fontsize: int = 18
    label_fontsize: int = 16
    tick_fontsize: int = 12
    legend_fontsize: int = 11
    trace_line_width: float = 0.6
    zoom_factor: float = 1.0 #modify if you want to zoom into the timeseries
    target_sampling_rate: int = 125 #modify based on your data
    freq_min: float = 0.5
    freq_max: float = 10.0 #modify based on your data

    # Spectrogram tuning
    spectrogram_vmin_percentile: float = 20.0
    spectrogram_vmax_percentile: float = 99.3
    spectrogram_title_y: float = 0.935
    spectrogram_top: float = 0.91
    spectrogram_bottom: float = 0.18
    spectrogram_hspace: float = 0.18
    spectrogram_xlabel_y: float = 0.145
    spectrogram_cbar_left: float = 0.28
    spectrogram_cbar_bottom: float = 0.085
    spectrogram_cbar_width: float = 0.44
    spectrogram_cbar_height: float = 0.020


CFG = QuietHourConfig(
    data_source="stream",
    station="COP",
    date_str="2009-02-19",
    selection_mode="secondary_metric",
    output_dir="quiet_hour_output",
)


# ============================================================
# STYLE HELPERS
# ============================================================

def apply_publication_style(cfg: QuietHourConfig) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": cfg.label_fontsize,
        "axes.labelsize": cfg.label_fontsize,
        "axes.titlesize": cfg.title_fontsize,
        "axes.titleweight": "bold",
        "xtick.labelsize": cfg.tick_fontsize,
        "ytick.labelsize": cfg.tick_fontsize,
        "legend.fontsize": cfg.legend_fontsize,
        "figure.dpi": cfg.figure_dpi,
        "savefig.dpi": cfg.figure_dpi,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax, xlabel=None, ylabel=None, title=None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="normal")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="normal")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="both", which="major", width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_figure(fig, output_base: str, cfg: QuietHourConfig) -> None:
    fig.savefig(f"{output_base}.png", dpi=cfg.figure_dpi, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label: str, fontsize: int) -> None:
    ax.text(
        0.012,
        0.93,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15),
    )


def add_corner_component_label(ax, label: str, fontsize: int) -> None:
    ax.text(
        0.985,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="right",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.15),
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def short_component_label(component: str) -> str:
    if component.endswith("Z"):
        return "Z"
    if component.endswith("N"):
        return "N"
    if component.endswith("E"):
        return "E"
    return component


def date_compact(date_str_dash: str) -> str:
    return UTCDateTime(date_str_dash).strftime("%Y%m%d")


def next_date_str(date_str_dash: str) -> str:
    return (UTCDateTime(date_str_dash) + 24 * 3600).strftime("%Y-%m-%d")


def record_hours_from_trace(trace) -> float:
    return max(0.0, (trace.stats.endtime - trace.stats.starttime) / 3600.0)


def hour_ticks(xmax: float) -> np.ndarray:
    return np.arange(0, int(np.floor(xmax)) + 1, 1)


def safe_decimate_to_target(trace, target_fs: float):
    fs = float(trace.stats.sampling_rate)
    if fs <= 0:
        return trace
    factor = int(fs / target_fs)
    if factor <= 1:
        return trace
    tr2 = trace.copy()
    tr2.decimate(factor, no_filter=True)
    return tr2


def validate_config(cfg: QuietHourConfig) -> None:
    if cfg.data_source.lower() not in {"local", "fdsn", "stream"}:
        raise ValueError("data_source must be 'local', 'fdsn', or 'stream'")

    UTCDateTime(cfg.date_str)

    if cfg.selection_mode not in {"first_metric", "secondary_metric"}:
        raise ValueError("selection_mode must be 'first_metric' or 'secondary_metric'")

    if cfg.data_source.lower() == "local" and not cfg.base_data_dir:
        raise ValueError("base_data_dir must be set for local mode")

    if cfg.data_source.lower() == "stream":
        has_user_stream = cfg.input_stream is not None and len(cfg.input_stream) > 0
        has_example_stream = cfg.example_stream_urls_current is not None
        if not has_user_stream and not has_example_stream:
            raise ValueError(
                "For stream mode, provide input_stream or example_stream_urls_current."
            )


def warn_missing_next_day(cfg: QuietHourConfig, source_label: str) -> None:
    warnings.warn(
        f"Next-day data are not available for {source_label} on {next_date_str(cfg.date_str)}. "
        f"Continuing without appended next-day data.",
        RuntimeWarning,
        stacklevel=2,
    )


# ============================================================
# DATA LOADING
# ============================================================

def merge_current_and_next_piece(current_trace, next_trace, append_next_day_seconds: int):
    next_start = next_trace.stats.starttime
    next_end = next_start + append_next_day_seconds
    next_piece = next_trace.copy().trim(next_start, next_end, pad=False)

    merged = Stream([current_trace.copy(), next_piece]).merge(
        method=1,
        fill_value="interpolate",
    )
    return merged[0]


def extract_stream_components(stream: Stream) -> Dict[str, object]:
    traces = {}
    for tr in stream:
        traces[tr.stats.channel] = tr.copy()
    return traces


def read_trace_for_local_date(
    cfg: QuietHourConfig,
    component: str,
    date_str_dash: str,
):
    data_dir = os.path.join(cfg.base_data_dir, cfg.month)
    file_path = os.path.join(
        data_dir,
        cfg.file_pattern.format(
            station=cfg.station,
            component=component,
            date=date_compact(date_str_dash),
        ),
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    return read(file_path)[0]


def load_local_traces(cfg: QuietHourConfig) -> Tuple[Dict[str, object], bool]:
    if cfg.channels is None:
        raise ValueError("channels must be set for local mode")

    traces = {}
    used_next_day = True
    next_day = next_date_str(cfg.date_str)

    for ch in cfg.channels:
        current_trace = read_trace_for_local_date(cfg, ch, cfg.date_str)

        try:
            next_trace = read_trace_for_local_date(cfg, ch, next_day)
            traces[ch] = merge_current_and_next_piece(
                current_trace=current_trace,
                next_trace=next_trace,
                append_next_day_seconds=cfg.append_next_day_seconds,
            )
        except FileNotFoundError:
            traces[ch] = current_trace
            used_next_day = False

    if not used_next_day:
        warn_missing_next_day(cfg, f"local data for station {cfg.station}")

    return traces, used_next_day


def load_example_streams(cfg: QuietHourConfig) -> Tuple[Stream, Optional[Stream]]:
    current_stream = Stream()
    next_stream = None

    for url in cfg.example_stream_urls_current or ():
        current_stream += read(url)

    if cfg.example_stream_urls_next_day:
        try:
            next_stream = Stream()
            for url in cfg.example_stream_urls_next_day:
                next_stream += read(url)
        except Exception:
            next_stream = None
            warn_missing_next_day(cfg, f"example stream data for station {cfg.station}")

    return current_stream, next_stream


def load_stream_traces(cfg: QuietHourConfig) -> Tuple[Dict[str, object], bool]:
    current_stream = cfg.input_stream
    next_stream = cfg.input_stream_next_day

    if current_stream is None or len(current_stream) == 0:
        current_stream, next_stream = load_example_streams(cfg)

    current_traces = extract_stream_components(current_stream)
    used_next_day = next_stream is not None and len(next_stream) > 0

    if not used_next_day:
        warn_missing_next_day(cfg, f"stream data for station {cfg.station}")
        return current_traces, False

    next_traces = extract_stream_components(next_stream)
    merged = {}

    for comp, current_trace in current_traces.items():
        if comp not in next_traces:
            merged[comp] = current_trace
            used_next_day = False
            continue

        merged[comp] = merge_current_and_next_piece(
            current_trace=current_trace,
            next_trace=next_traces[comp],
            append_next_day_seconds=cfg.append_next_day_seconds,
        )

    if not used_next_day:
        warn_missing_next_day(cfg, f"stream data for station {cfg.station}")

    return merged, used_next_day


def discover_fdsn_three_components(
    client: Client,
    cfg: QuietHourConfig,
    query_date_str: str,
) -> Tuple[str, str, Tuple[str, str, str]]:
    start = UTCDateTime(query_date_str)
    end = start + 24 * 3600

    inventory = client.get_stations(
        network=cfg.network,
        station=cfg.station,
        location="*" if cfg.location is None else cfg.location,
        channel="*",
        starttime=start,
        endtime=end,
        level="channel",
    )

    candidates = []

    for net in inventory:
        for sta in net:
            grouped = {}
            for cha in sta:
                loc = cha.location_code or ""
                code = cha.code

                if len(code) < 3:
                    continue

                suffix = code[-1]
                prefix = code[:2]

                if suffix not in {"Z", "N", "E"}:
                    continue

                grouped.setdefault((loc, prefix), set()).add(code)

            for (loc, prefix), codes in grouped.items():
                z = f"{prefix}Z"
                n = f"{prefix}N"
                e = f"{prefix}E"
                if {z, n, e}.issubset(codes):
                    candidates.append((loc, prefix, (z, n, e)))

    if not candidates:
        raise RuntimeError(
            f"No complete Z/N/E triplet found for {cfg.network}.{cfg.station} on {query_date_str}"
        )

    for preferred_prefix in cfg.channel_band_priority:
        for loc, prefix, triplet in candidates:
            if prefix == preferred_prefix:
                return loc, prefix, triplet

    return candidates[0]


def _download_one_channel(
    client: Client,
    cfg: QuietHourConfig,
    location: str,
    channel: str,
    start: UTCDateTime,
    end: UTCDateTime,
):
    print(
        f"Requesting public data: client={cfg.fdsn_client}, "
        f"net={cfg.network}, sta={cfg.station}, loc={location or '--'}, "
        f"cha={channel}, start={start}, end={end}"
    )

    st = client.get_waveforms(
        network=cfg.network,
        station=cfg.station,
        location=location,
        channel=channel,
        starttime=start,
        endtime=end,
        attach_response=False,
    )
    st.merge(method=1, fill_value="interpolate")

    if len(st) == 0:
        raise RuntimeError(f"No waveform returned for channel {channel}")

    tr = st[0]

    if cfg.save_downloaded_waveforms:
        os.makedirs(cfg.output_dir, exist_ok=True)
        fname = (
            f"{cfg.network}.{cfg.station}.{location or '--'}."
            f"{channel}.{start.strftime('%Y%m%d')}.{cfg.downloaded_format.lower()}"
        )
        out_path = os.path.join(cfg.output_dir, fname)
        tr.write(out_path, format=cfg.downloaded_format)
        print(f"Saved downloaded waveform: {out_path}")

    return tr


def load_fdsn_traces(cfg: QuietHourConfig) -> Tuple[Dict[str, object], bool]:
    client = Client(cfg.fdsn_client)
    current_start = UTCDateTime(cfg.date_str)
    current_end = current_start + 24 * 3600
    next_start = current_end
    next_end = next_start + cfg.append_next_day_seconds

    if cfg.channels is None or cfg.location is None:
        location, prefix, channels = discover_fdsn_three_components(client, cfg, cfg.date_str)
        print(
            f"Auto-selected public data triplet: "
            f"{cfg.network}.{cfg.station}.{location or '--'}.{prefix}[ZNE]"
        )
    else:
        location = cfg.location
        channels = cfg.channels

    traces = {}
    used_next_day = True

    for ch in channels:
        current_trace = _download_one_channel(
            client=client,
            cfg=cfg,
            location=location,
            channel=ch,
            start=current_start,
            end=current_end,
        )

        try:
            next_trace = _download_one_channel(
                client=client,
                cfg=cfg,
                location=location,
                channel=ch,
                start=next_start,
                end=next_end,
            )
            traces[ch] = merge_current_and_next_piece(
                current_trace=current_trace,
                next_trace=next_trace,
                append_next_day_seconds=cfg.append_next_day_seconds,
            )
        except Exception:
            traces[ch] = current_trace
            used_next_day = False

    if not used_next_day:
        warn_missing_next_day(cfg, f"FDSN data for station {cfg.network}.{cfg.station}")

    return traces, used_next_day


def load_traces(cfg: QuietHourConfig) -> Tuple[Dict[str, object], bool]:
    mode = cfg.data_source.lower()

    if mode == "local":
        return load_local_traces(cfg)
    if mode == "fdsn":
        return load_fdsn_traces(cfg)
    if mode == "stream":
        return load_stream_traces(cfg)

    raise ValueError("data_source must be 'local', 'fdsn', or 'stream'")


# ============================================================
# QUIET-HOUR METRICS
# ============================================================

def get_component_roles(components) -> Tuple[str, str, str]:
    z = next((c for c in components if c.endswith("Z")), None)
    n = next((c for c in components if c.endswith("N")), None)
    e = next((c for c in components if c.endswith("E")), None)
    if z is None or n is None or e is None:
        raise ValueError(f"Need one Z, one N, and one E component. Got: {components}")
    return z, n, e


def trim_components_to_common_range(traces: Dict[str, object]) -> Dict[str, object]:
    common_start = max(tr.stats.starttime for tr in traces.values())
    common_end = min(tr.stats.endtime for tr in traces.values())

    out = {}
    for comp, tr in traces.items():
        out[comp] = tr.copy().trim(common_start, common_end, pad=False)

    min_npts = min(tr.stats.npts for tr in out.values())
    for comp in out:
        out[comp].data = out[comp].data[:min_npts]
        out[comp].stats.npts = min_npts

    return out


def compute_window_std(
    traces: Dict[str, object],
    component_z: str,
    component_n: str,
    component_e: str,
    window_length: int,
    step: int,
    valid_day_end: UTCDateTime,
) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float], List[float]]:
    starttime = traces[component_z].stats.starttime
    endtime = traces[component_z].stats.endtime
    sample_interval = 1.0 / traces[component_z].stats.sampling_rate

    time_list = []
    std_z_list = []
    std_n_list = []
    std_e_list = []
    std_sum_list = []

    current_start = starttime

    while current_start + window_length - sample_interval <= endtime:
        if current_start >= valid_day_end:
            break

        current_end = current_start + window_length

        z_win = traces[component_z].slice(current_start, current_end).data
        n_win = traces[component_n].slice(current_start, current_end).data
        e_win = traces[component_e].slice(current_start, current_end).data

        std_z = float(np.std(z_win))
        std_n = float(np.std(n_win))
        std_e = float(np.std(e_win))
        std_sum = std_z + std_n + std_e

        time_list.append(pd.Timestamp(current_start.datetime))
        std_z_list.append(std_z)
        std_n_list.append(std_n)
        std_e_list.append(std_e)
        std_sum_list.append(std_sum)

        current_start += step

    return time_list, std_z_list, std_n_list, std_e_list, std_sum_list


def compute_secondary_metric(
    std_sum_list: List[float],
    window_length: int,
    step: int,
) -> Tuple[np.ndarray, int]:
    num_steps = int(window_length / step)
    rolling_sums = pd.Series(std_sum_list).rolling(window=num_steps).sum().to_numpy()
    return rolling_sums, num_steps


def select_quiet_window_direct(
    time_list: List[pd.Timestamp],
    std_sum_list: List[float],
    window_length: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, float, int, int, Optional[np.ndarray]]:
    if not time_list:
        raise ValueError("No valid windows were computed.")

    min_index = int(np.argmin(std_sum_list))
    best_start = pd.to_datetime(time_list[min_index])
    best_end = best_start + pd.Timedelta(seconds=window_length)
    quiet_metric = float(std_sum_list[min_index])

    return best_start, best_end, quiet_metric, min_index, min_index, None


def select_quiet_window_secondary(
    time_list: List[pd.Timestamp],
    std_sum_list: List[float],
    window_length: int,
    step: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, float, int, int, np.ndarray]:
    if not time_list:
        raise ValueError("No valid windows were computed.")

    rolling_sums, num_steps = compute_secondary_metric(
        std_sum_list=std_sum_list,
        window_length=window_length,
        step=step,
    )

    min_index = int(np.nanargmin(rolling_sums))
    if min_index < num_steps - 1:
        raise ValueError("Quiet window selection failed because minimum rolling window is incomplete.")

    start_index = min_index - num_steps + 1
    best_start = pd.to_datetime(time_list[start_index])
    best_end = best_start + pd.Timedelta(seconds=window_length)
    quiet_metric = float(rolling_sums[min_index])

    return best_start, best_end, quiet_metric, start_index, min_index, rolling_sums


def select_quiet_window(
    time_list: List[pd.Timestamp],
    std_sum_list: List[float],
    window_length: int,
    step: int,
    selection_mode: str,
):
    if selection_mode == "first_metric":
        return select_quiet_window_direct(time_list, std_sum_list, window_length)
    if selection_mode == "secondary_metric":
        return select_quiet_window_secondary(time_list, std_sum_list, window_length, step)
    raise ValueError("selection_mode must be 'first_metric' or 'secondary_metric'")


# ============================================================
# QC PLOTS
# ============================================================

def _compute_spectrogram_bundle(trace, cfg: QuietHourConfig):
    tr = safe_decimate_to_target(trace.copy(), cfg.target_sampling_rate)
    f, t, sxx = spectrogram(
        tr.data.astype(np.float32),
        fs=tr.stats.sampling_rate,
        nperseg=512,
        noverlap=256,
    )
    sxx_db = 10 * np.log10(sxx + 1e-20)
    return f, t / 3600.0, sxx_db


def _spectrogram_limits(specs, cfg: QuietHourConfig):
    pooled = []
    for _, f, _, sxx_db in specs:
        band = (f >= 1.5) & (f <= 10.0)
        if np.any(band):
            pooled.append(sxx_db[band, :].ravel())
    if not pooled:
        pooled = [s[3].ravel() for s in specs]
    pooled_all = np.concatenate(pooled)
    vmin = float(np.percentile(pooled_all, cfg.spectrogram_vmin_percentile))
    vmax = float(np.percentile(pooled_all, cfg.spectrogram_vmax_percentile))
    return vmin, vmax


def plot_std_and_metric(
    cfg: QuietHourConfig,
    traces: Dict[str, object],
    time_list: List[pd.Timestamp],
    std_z_list: List[float],
    std_n_list: List[float],
    std_e_list: List[float],
    std_sum_list: List[float],
    quiet_window_df: pd.DataFrame,
    rolling_sums: Optional[np.ndarray],
):
    apply_publication_style(cfg)
    z, n, e = get_component_roles(tuple(traces.keys()))

    starttime = traces[z].stats.starttime
    std_times = [(UTCDateTime(t) - starttime) / 3600.0 for t in time_list]
    xmax = min(24.0, record_hours_from_trace(traces[z]))

    best_start = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_start"])
    best_end = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_end"])
    quiet_start_hr = (UTCDateTime(best_start.to_pydatetime()) - starttime) / 3600.0
    quiet_end_hr = (UTCDateTime(best_end.to_pydatetime()) - starttime) / 3600.0

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(std_times, std_z_list, label="Z", color="#1f77b4")
    ax.plot(std_times, std_n_list, label="N", color="#2ca02c")
    ax.plot(std_times, std_e_list, label="E", color="#ff7f0e")
    ax.plot(std_times, std_sum_list, label="Total STD", linestyle="--", color="black", linewidth=1.8)
    ax.axvspan(quiet_start_hr, quiet_end_hr, color="red", alpha=0.12, label="Quiet Window")

    ax.set_xlim(0, xmax)
    ax.set_xticks(hour_ticks(xmax))
    ax.grid(alpha=0.25)
    style_axis(ax, xlabel="Hour (UTC)", ylabel="STD", title=f"STD - {cfg.station} {cfg.date_str}")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    save_figure(fig, os.path.join(cfg.output_dir, "quiet_hour_std"), cfg)

    if rolling_sums is not None:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(std_times, rolling_sums, color="#8e44ad", linewidth=1.8, label="Secondary Metric")
        ax.axvspan(quiet_start_hr, quiet_end_hr, color="red", alpha=0.12, label="Quiet Window")

        ax.set_xlim(0, xmax)
        ax.set_xticks(hour_ticks(xmax))
        ax.grid(alpha=0.25)
        style_axis(ax, xlabel="Hour (UTC)", ylabel="Secondary Metric", title=f"Secondary Metric - {cfg.station} {cfg.date_str}")
        ax.legend(frameon=True, loc="upper left")
        fig.tight_layout()
        save_figure(fig, os.path.join(cfg.output_dir, "quiet_hour_secondary_metric"), cfg)


def plot_quiet_hour_timeseries(
    cfg: QuietHourConfig,
    traces: Dict[str, object],
    quiet_window_df: pd.DataFrame,
):
    apply_publication_style(cfg)

    z, n, e = get_component_roles(tuple(traces.keys()))
    ordered = [z, n, e]
    starttime = traces[z].stats.starttime
    xmax = min(24.0, record_hours_from_trace(traces[z]))

    best_start = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_start"])
    best_end = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_end"])
    quiet_start_hr = (UTCDateTime(best_start.to_pydatetime()) - starttime) / 3600.0
    quiet_end_hr = (UTCDateTime(best_end.to_pydatetime()) - starttime) / 3600.0

    fig, axs = plt.subplots(3, 1, figsize=(14, 8.8), sharex=True)
    fig.suptitle(
        f"Timeseries - {cfg.station} {cfg.date_str}",
        fontsize=cfg.title_fontsize,
        fontweight="bold",
        y=0.975,
    )

    max_ampl = 0.0
    cache = {}

    for comp in ordered:
        tr_dec = safe_decimate_to_target(traces[comp].copy(), cfg.target_sampling_rate)
        data = tr_dec.data.astype(np.float32)
        t_hours = tr_dec.times() / 3600.0
        cache[comp] = (t_hours, data)
        if len(data) > 0:
            max_ampl = max(max_ampl, float(np.nanmax(np.abs(data))))

    max_ampl = max(max_ampl, 1.0)

    for i, comp in enumerate(ordered):
        t_hours, data = cache[comp]
        axs[i].plot(t_hours, data, color="black", linewidth=cfg.trace_line_width)
        axs[i].axvline(quiet_start_hr, color="red", linestyle="--", linewidth=1.1, alpha=0.9)
        axs[i].axvline(quiet_end_hr, color="red", linestyle="--", linewidth=1.1, alpha=0.9)
        axs[i].set_ylim(-max_ampl * cfg.zoom_factor, max_ampl * cfg.zoom_factor)
        style_axis(axs[i], ylabel="Amplitude")
        add_panel_label(axs[i], f"({chr(97 + i)})", cfg.tick_fontsize + 1)
        add_corner_component_label(axs[i], short_component_label(comp), cfg.label_fontsize)

    axs[-1].set_xlim(0, xmax)
    axs[-1].set_xticks(hour_ticks(xmax))
    style_axis(axs[-1], xlabel="Hour (UTC)")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.10, hspace=0.10)
    save_figure(fig, os.path.join(cfg.output_dir, "quiet_hour_timeseries"), cfg)


def plot_quiet_hour_spectrogram(
    cfg: QuietHourConfig,
    traces: Dict[str, object],
    quiet_window_df: pd.DataFrame,
):
    apply_publication_style(cfg)

    z, n, e = get_component_roles(tuple(traces.keys()))
    ordered = [z, n, e]
    starttime = traces[z].stats.starttime
    xmax = min(24.0, record_hours_from_trace(traces[z]))

    best_start = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_start"])
    best_end = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_end"])
    quiet_start_hr = (UTCDateTime(best_start.to_pydatetime()) - starttime) / 3600.0
    quiet_end_hr = (UTCDateTime(best_end.to_pydatetime()) - starttime) / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.6), sharex=True)

    specs = [(comp, *_compute_spectrogram_bundle(traces[comp].copy(), cfg)) for comp in ordered]
    vmin_global, vmax_global = _spectrogram_limits(specs, cfg)

    im = None
    for i, (comp, f, t, sxx_db) in enumerate(specs):
        im = axes[i].imshow(
            sxx_db,
            aspect="auto",
            extent=[t[0], t[-1], f[0], f[-1]],
            origin="lower",
            cmap="viridis",
            vmin=vmin_global,
            vmax=vmax_global,
        )
        axes[i].axvline(quiet_start_hr, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        axes[i].axvline(quiet_end_hr, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        axes[i].set_ylim(cfg.freq_min, cfg.freq_max)
        style_axis(axes[i], ylabel="Freq (Hz)")
        add_panel_label(axes[i], f"({chr(97 + i)})", cfg.tick_fontsize + 1)
        add_corner_component_label(axes[i], short_component_label(comp), cfg.label_fontsize)

    axes[-1].set_xlim(0, xmax)
    axes[-1].set_xticks(hour_ticks(xmax))
    axes[-1].set_xlabel("")

    fig.suptitle(
        f"Spectrogram - {cfg.station} {cfg.date_str}",
        fontsize=cfg.title_fontsize,
        fontweight="bold",
        y=cfg.spectrogram_title_y,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=cfg.spectrogram_top,
        bottom=cfg.spectrogram_bottom,
        hspace=cfg.spectrogram_hspace,
    )

    fig.text(0.50, cfg.spectrogram_xlabel_y, "Hour (UTC)", ha="center", va="center", fontsize=cfg.label_fontsize)

    cax = fig.add_axes([
        cfg.spectrogram_cbar_left,
        cfg.spectrogram_cbar_bottom,
        cfg.spectrogram_cbar_width,
        cfg.spectrogram_cbar_height,
    ])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Power (dB)", fontweight="normal", fontsize=cfg.label_fontsize, labelpad=2)

    save_figure(fig, os.path.join(cfg.output_dir, "quiet_hour_spectrogram"), cfg)


def plot_quiet_hour_combined_qc(
    cfg: QuietHourConfig,
    traces: Dict[str, object],
    time_list: List[pd.Timestamp],
    std_z_list: List[float],
    std_n_list: List[float],
    std_e_list: List[float],
    std_sum_list: List[float],
    quiet_window_df: pd.DataFrame,
):
    apply_publication_style(cfg)

    z, n, e = get_component_roles(tuple(traces.keys()))
    ordered = [z, n, e]
    starttime = traces[z].stats.starttime
    xmax = min(24.0, record_hours_from_trace(traces[z]))

    best_start = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_start"])
    best_end = pd.to_datetime(quiet_window_df.loc[0, "quiet_window_end"])
    quiet_start_hr = (UTCDateTime(best_start.to_pydatetime()) - starttime) / 3600.0
    quiet_end_hr = (UTCDateTime(best_end.to_pydatetime()) - starttime) / 3600.0
    std_times = [(UTCDateTime(t) - starttime) / 3600.0 for t in time_list]

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(15, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2]},
    )

    axes[0].plot(std_times, std_z_list, label="Z", color="#1f77b4")
    axes[0].plot(std_times, std_n_list, label="N", color="#2ca02c")
    axes[0].plot(std_times, std_e_list, label="E", color="#ff7f0e")
    axes[0].plot(std_times, std_sum_list, label="Total STD", linestyle="--", color="black", linewidth=1.8)
    axes[0].axvspan(quiet_start_hr, quiet_end_hr, color="red", alpha=0.12, label="Quiet Window")
    axes[0].grid(alpha=0.25)
    style_axis(axes[0], ylabel="STD", title=f"STD - {cfg.station} {cfg.date_str}")
    axes[0].legend(frameon=True, loc="upper left")

    max_ampl = 0.0
    ts_cache = {}
    for comp in ordered:
        tr_dec = safe_decimate_to_target(traces[comp].copy(), cfg.target_sampling_rate)
        data = tr_dec.data.astype(np.float32)
        t_hours = tr_dec.times() / 3600.0
        ts_cache[comp] = (t_hours, data)
        if len(data) > 0:
            max_ampl = max(max_ampl, float(np.nanmax(np.abs(data))))
    max_ampl = max(max_ampl, 1.0)

    for i, comp in enumerate(ordered):
        t_hours, data = ts_cache[comp]
        axes[i + 1].plot(t_hours, data, color="black", linewidth=cfg.trace_line_width)
        axes[i + 1].axvline(quiet_start_hr, color="red", linestyle="--", linewidth=1.1, alpha=0.9)
        axes[i + 1].axvline(quiet_end_hr, color="red", linestyle="--", linewidth=1.1, alpha=0.9)
        axes[i + 1].set_ylim(-max_ampl * cfg.zoom_factor, max_ampl * cfg.zoom_factor)
        style_axis(axes[i + 1], ylabel="Amplitude")
        add_panel_label(axes[i + 1], f"({chr(97 + i)})", cfg.tick_fontsize + 1)
        add_corner_component_label(axes[i + 1], short_component_label(comp), cfg.label_fontsize)

    specs = [(comp, *_compute_spectrogram_bundle(traces[comp].copy(), cfg)) for comp in ordered]
    vmin_global, vmax_global = _spectrogram_limits(specs, cfg)

    for i, (comp, f, t, sxx_db) in enumerate(specs):
        axes[i + 4].imshow(
            sxx_db,
            aspect="auto",
            extent=[t[0], t[-1], f[0], f[-1]],
            origin="lower",
            cmap="viridis",
            vmin=vmin_global,
            vmax=vmax_global,
        )
        axes[i + 4].axvline(quiet_start_hr, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        axes[i + 4].axvline(quiet_end_hr, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        axes[i + 4].set_ylim(cfg.freq_min, cfg.freq_max)
        style_axis(axes[i + 4], ylabel="Freq (Hz)")
        add_panel_label(axes[i + 4], f"({chr(97 + i)})", cfg.tick_fontsize + 1)
        add_corner_component_label(axes[i + 4], short_component_label(comp), cfg.label_fontsize)

    axes[-1].set_xlim(0, xmax)
    axes[-1].set_xticks(hour_ticks(xmax))
    style_axis(axes[-1], xlabel="Hour (UTC)")

    fig.subplots_adjust(left=0.09, right=0.985, top=0.97, bottom=0.05, hspace=0.12)
    save_figure(fig, os.path.join(cfg.output_dir, "quiet_hour_combined_qc"), cfg)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = CFG
    validate_config(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    print("Output directory:", os.path.abspath(cfg.output_dir))

    traces, used_next_day = load_traces(cfg)
    traces = trim_components_to_common_range(traces)

    z, n, e = get_component_roles(tuple(traces.keys()))
    day_start = UTCDateTime(cfg.date_str)
    valid_day_end = day_start + 24 * 3600

    time_list, std_z_list, std_n_list, std_e_list, std_sum_list = compute_window_std(
        traces=traces,
        component_z=z,
        component_n=n,
        component_e=e,
        window_length=cfg.window_length_sec,
        step=cfg.step_sec,
        valid_day_end=valid_day_end,
    )

    best_start, best_end, quiet_metric, start_index, end_index, rolling_sums = select_quiet_window(
        time_list=time_list,
        std_sum_list=std_sum_list,
        window_length=cfg.window_length_sec,
        step=cfg.step_sec,
        selection_mode=cfg.selection_mode,
    )

    quiet_window_df = pd.DataFrame([{
        "quiet_window_start": best_start,
        "quiet_window_end": best_end,
        "quiet_metric": quiet_metric,
        "start_index": start_index,
        "end_index": end_index,
        "selection_mode": cfg.selection_mode,
        "used_next_day_data": used_next_day,
    }])
    quiet_window_path = os.path.join(cfg.output_dir, "quiet_hour_selection.csv")
    quiet_window_df.to_csv(quiet_window_path, index=False)

    rolling_column = rolling_sums if rolling_sums is not None else np.full(len(time_list), np.nan)
    std_df = pd.DataFrame({
        "Time_UTC": time_list,
        "STD_Z": std_z_list,
        "STD_N": std_n_list,
        "STD_E": std_e_list,
        "STD_SUM": std_sum_list,
        "Secondary_Rolling_Sum": rolling_column,
        "Best_Window": [start_index <= i <= end_index for i in range(len(time_list))],
    })
    std_windows_path = os.path.join(cfg.output_dir, "quiet_hour_std_windows.csv")
    std_df.to_csv(std_windows_path, index=False)

    print("Selected quiet window:")
    print(quiet_window_df)
    print("Saved:", os.path.abspath(quiet_window_path))
    print("Saved:", os.path.abspath(std_windows_path))

    plot_std_and_metric(
        cfg,
        traces,
        time_list,
        std_z_list,
        std_n_list,
        std_e_list,
        std_sum_list,
        quiet_window_df,
        rolling_sums,
    )
    plot_quiet_hour_timeseries(cfg, traces, quiet_window_df)
    plot_quiet_hour_spectrogram(cfg, traces, quiet_window_df)
    plot_quiet_hour_combined_qc(
        cfg,
        traces,
        time_list,
        std_z_list,
        std_n_list,
        std_e_list,
        std_sum_list,
        quiet_window_df,
    )


if __name__ == "__main__":
    main()
