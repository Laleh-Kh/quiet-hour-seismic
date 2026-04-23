from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class QuietHourConfig:
    """
    Quiet-hour selection from three built-in Obspy example waveform files.

    Method:
    1. Compute the standard deviation for Z, N, and E in each window.
    2. Sum those three values to get ``std_sum``.
    3. Smooth ``std_sum`` with a rolling sum over one hour window.
    4. Pick the quiet hour from the minimum of ``smoothed_std_sum``.
    """

    station: str = "COP"
    channels: tuple[str, str, str] = ("BHZ", "BHN", "BHE")
    date_str: str = "2009-02-19"
    example_stream_urls_current: tuple[str, str, str] = (
        "https://examples.obspy.org/COP.BHE.DK.2009.050",
        "https://examples.obspy.org/COP.BHN.DK.2009.050",
        "https://examples.obspy.org/COP.BHZ.DK.2009.050",
    )
    example_stream_urls_next_day: tuple[str, str, str] = (
        "example.BHE",
        "examples.BHN",
        "examples.BHZ",
    )

    window_length_sec = 3600
    step_sec = 60
    append_next_day_seconds = 3600

    output_dir = os.path.join(SCRIPT_DIR, "quiet_hour_output")
    figure_dpi = 300


CFG = QuietHourConfig()


def next_date_str(date_str_dash: str) -> str:
    return (UTCDateTime(date_str_dash) + 24 * 3600).strftime("%Y-%m-%d")


def hour_ticks(xmax: float) -> np.ndarray:
    return np.arange(0, int(np.floor(xmax)) + 1, 1)


def validate_config(cfg: QuietHourConfig) -> None:
    UTCDateTime(cfg.date_str)

    if len(cfg.channels) != 3:
        raise ValueError("channels must contain exactly three components: Z, N, and E.")

    if cfg.window_length_sec <= 0 or cfg.step_sec <= 0:
        raise ValueError("window_length_sec and step_sec must be positive.")

    if cfg.window_length_sec % cfg.step_sec != 0:
        raise ValueError("window_length_sec must be divisible by step_sec.")


def merge_current_and_next_piece(current_trace, next_trace, append_next_day_seconds: int):
    next_start = next_trace.stats.starttime
    next_end = next_start + append_next_day_seconds
    next_piece = next_trace.copy().trim(next_start, next_end, pad=False)

    merged = Stream([current_trace.copy(), next_piece]).merge(
        method=1,
        fill_value="interpolate",
    )
    return merged[0]


def load_example_traces(cfg: QuietHourConfig):
    traces = {}
    used_next_day = True
    next_day = next_date_str(cfg.date_str)

    for current_url, next_url in zip(cfg.example_stream_urls_current, cfg.example_stream_urls_next_day):
        current_trace = read(current_url)[0]
        channel = current_trace.stats.channel

        try:
            next_trace = read(next_url)[0]
            traces[channel] = merge_current_and_next_piece(
                current_trace=current_trace,
                next_trace=next_trace,
                append_next_day_seconds=cfg.append_next_day_seconds,
            )
        except Exception:
            traces[channel] = current_trace
            used_next_day = False

    if not used_next_day:
        warnings.warn(
            f"Next-day data are not available for {cfg.station} on {next_day}. "
            "Continuing with the current day only.",
            RuntimeWarning,
            stacklevel=2,
        )

    return traces, used_next_day


def get_component_roles(components: tuple[str, str, str]) -> tuple[str, str, str]:
    z = next((c for c in components if c.endswith("Z")), None)
    n = next((c for c in components if c.endswith("N")), None)
    e = next((c for c in components if c.endswith("E")), None)
    if z is None or n is None or e is None:
        raise ValueError(f"Need one Z, one N, and one E component. Got: {components}")
    return z, n, e


def trim_components_to_common_range(traces):
    common_start = max(tr.stats.starttime for tr in traces.values())
    common_end = min(tr.stats.endtime for tr in traces.values())

    trimmed = {}
    for component, trace in traces.items():
        trimmed[component] = trace.copy().trim(common_start, common_end, pad=False)

    min_npts = min(trace.stats.npts for trace in trimmed.values())
    for trace in trimmed.values():
        trace.data = trace.data[:min_npts]
        trace.stats.npts = min_npts

    return trimmed


def compute_window_metrics(
    traces,
    component_z: str,
    component_n: str,
    component_e: str,
    window_length_sec: int,
    step_sec: int,
    valid_day_end: UTCDateTime,
):
    starttime = traces[component_z].stats.starttime
    endtime = traces[component_z].stats.endtime
    sample_interval = 1.0 / traces[component_z].stats.sampling_rate

    window_starts = []
    std_z = []
    std_n = []
    std_e = []
    std_sum = []

    current_start = starttime

    while current_start + window_length_sec - sample_interval <= endtime:
        if current_start >= valid_day_end:
            break

        current_end = current_start + window_length_sec

        z_win = traces[component_z].slice(current_start, current_end).data
        n_win = traces[component_n].slice(current_start, current_end).data
        e_win = traces[component_e].slice(current_start, current_end).data

        z_std = float(np.std(z_win))
        n_std = float(np.std(n_win))
        e_std = float(np.std(e_win))

        window_starts.append(pd.Timestamp(current_start.datetime))
        std_z.append(z_std)
        std_n.append(n_std)
        std_e.append(e_std)
        std_sum.append(z_std + n_std + e_std)

        current_start += step_sec

    smoothed_std_sum = (
        pd.Series(std_sum)
        .rolling(window=window_length_sec // step_sec, min_periods=window_length_sec // step_sec)
        .sum()
        .to_numpy()
    )

    return pd.DataFrame(
        {
            "window_start_utc": window_starts,
            "std_z": std_z,
            "std_n": std_n,
            "std_e": std_e,
            "std_sum": std_sum,
            "smoothed_std_sum": smoothed_std_sum,
        }
    )


def select_quiet_hour(metrics_df: pd.DataFrame, window_length_sec: int) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    if metrics_df.empty:
        raise ValueError("No valid windows were computed.")

    smoothed_values = metrics_df["smoothed_std_sum"].to_numpy()
    if np.all(np.isnan(smoothed_values)):
        raise ValueError("smoothed_std_sum could not be computed.")

    quiet_end_index = int(np.nanargmin(smoothed_values))
    first_valid_index = metrics_df["smoothed_std_sum"].first_valid_index()
    window_size = int(first_valid_index) + 1
    quiet_start_index = quiet_end_index - window_size + 1

    quiet_start = pd.to_datetime(metrics_df.loc[quiet_start_index, "window_start_utc"])
    quiet_end = quiet_start + pd.Timedelta(seconds=window_length_sec)
    return quiet_start, quiet_end, quiet_start_index


def plot_metrics(cfg: QuietHourConfig, metrics_df: pd.DataFrame, traces, quiet_start, quiet_end) -> None:
    z, _, _ = get_component_roles(tuple(traces.keys()))
    trace_start = traces[z].stats.starttime
    xmax = min(24.0, (traces[z].stats.endtime - traces[z].stats.starttime) / 3600.0)

    x_hours = [
        (UTCDateTime(timestamp.to_pydatetime()) - trace_start) / 3600.0
        for timestamp in metrics_df["window_start_utc"]
    ]
    quiet_start_hr = (UTCDateTime(quiet_start.to_pydatetime()) - trace_start) / 3600.0
    quiet_end_hr = (UTCDateTime(quiet_end.to_pydatetime()) - trace_start) / 3600.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )

    line_z = ax_top.plot(x_hours, metrics_df["std_z"], color="#1f77b4", label="Z")[0]
    line_n = ax_top.plot(x_hours, metrics_df["std_n"], color="#2ca02c", label="N")[0]
    line_e = ax_top.plot(x_hours, metrics_df["std_e"], color="#ff7f0e", label="E")[0]
    line_sum = ax_top.plot(
        x_hours,
        metrics_df["std_sum"],
        color="#222222",
        linestyle="--",
        linewidth=1.6,
        alpha=0.8,
        label="STD Sum",
        zorder=2,
    )[0]
    ax_top.grid(alpha=0.25)
    ax_top.legend(
        [line_z, line_n, line_e, line_sum],
        [
            line_z.get_label(),
            line_n.get_label(),
            line_e.get_label(),
            line_sum.get_label(),
        ],
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.95,
    )
    ax_top.set_ylabel("STD")

    line_smoothed = ax_bottom.plot(
        x_hours,
        metrics_df["smoothed_std_sum"],
        color="#8e44ad",
        linewidth=2.4,
        label="Rolling Sum of STD Sum",
        zorder=4,
    )[0]
    quiet_patch_bottom = ax_bottom.axvspan(
        quiet_start_hr,
        quiet_end_hr,
        color="red",
        alpha=0.10,
        label="Selected Quiet Hour",
    )
    ax_bottom.set_xlim(0, xmax)
    ax_bottom.set_xticks(hour_ticks(xmax))
    ax_bottom.set_xlabel("Hour (UTC)")
    ax_bottom.set_ylabel("Rolling Sum of STD Sum", color="#8e44ad")
    ax_bottom.tick_params(axis="y", colors="#8e44ad")
    ax_bottom.spines["left"].set_color("#8e44ad")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.legend(
        [line_smoothed, quiet_patch_bottom],
        [
            line_smoothed.get_label(),
            quiet_patch_bottom.get_label(),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    ax_top.set_title(f"Quiet-hour selection for {cfg.station} on {cfg.date_str}")

    fig.tight_layout()
    output_base = os.path.join(cfg.output_dir, "quiet_hour_metrics")
    fig.savefig(f"{output_base}.png", dpi=cfg.figure_dpi, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = CFG
    validate_config(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)

    traces, used_next_day = load_example_traces(cfg)
    traces = trim_components_to_common_range(traces)

    z, n, e = get_component_roles(tuple(traces.keys()))
    valid_day_end = UTCDateTime(cfg.date_str) + 24 * 3600

    metrics_df = compute_window_metrics(
        traces=traces,
        component_z=z,
        component_n=n,
        component_e=e,
        window_length_sec=cfg.window_length_sec,
        step_sec=cfg.step_sec,
        valid_day_end=valid_day_end,
    )

    quiet_start, quiet_end, quiet_start_index = select_quiet_hour(
        metrics_df=metrics_df,
        window_length_sec=cfg.window_length_sec,
    )

    quiet_end_index = quiet_start_index + (cfg.window_length_sec // cfg.step_sec) - 1
    metrics_df["selected_quiet_hour"] = False
    metrics_df.loc[quiet_start_index:quiet_end_index, "selected_quiet_hour"] = True

    summary_df = pd.DataFrame(
        [
            {
                "quiet_hour_start": quiet_start,
                "quiet_hour_end": quiet_end,
                "std_sum_at_start": metrics_df.loc[quiet_start_index, "std_sum"],
                "minimum_smoothed_std_sum": metrics_df.loc[quiet_end_index, "smoothed_std_sum"],
                "used_next_day_data": used_next_day,
            }
        ]
    )

    metrics_path = os.path.join(cfg.output_dir, "quiet_hour_metrics.csv")
    summary_path = os.path.join(cfg.output_dir, "quiet_hour_selection.csv")
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Selected quiet hour:")
    print(summary_df)
    print("Saved:", os.path.abspath(summary_path))
    print("Saved:", os.path.abspath(metrics_path))

    plot_metrics(
        cfg=cfg,
        metrics_df=metrics_df,
        traces=traces,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
    )


if __name__ == "__main__":
    main()
