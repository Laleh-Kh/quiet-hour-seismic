"""
select_quiet_hour.py

Select the quietest fixed-length time window in continuous 3-component seismic data
using a sliding-window standard deviation approach.

Outputs:
- Per-day CSV files of rolling standard deviations
- Monthly summary CSV files of selected quiet windows
- QC plots showing rolling STD, time series, and spectrograms

This workflow was originally developed to support HVSR preprocessing in
continuous passive seismic data, but it is general and can be used for other
continuous seismic applications.

Author: Laleh Khadangi
"""

from __future__ import annotations

import os
import glob
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from scipy.signal import spectrogram


# ============================================================
# USER CONFIGURATION
# ============================================================

@dataclass
class Config:
    base_data_dir: str
    base_output_dir: str
    stations: List[str]
    months: List[str]
    components: List[str]
    file_pattern: str
    date_extractor: str = "underscore_last"
    window_length: int = 3600           # seconds
    step: int = 60                      # seconds
    zoom_factor: float = 0.05
    freq_min: float = 0.5
    freq_max: float = 25.0
    target_sampling_rate: int = 125
    save_figures: bool = True


CONFIG = Config(
    base_data_dir="PATH_TO_DATA",
    base_output_dir="PATH_TO_OUTPUT",

    # Example:
    # stations=["F0208", "F0203"]
    stations=["STATION_NAME"],

    # Example:
    # months=["Jan", "Feb", "Mar"]
    months=["MONTH_NAME"],

    # Component names must end with Z, N, E
    # Examples:
    # ["DPZ", "DPN", "DPE"]
    # ["Z", "N", "E"]
    components=["Z", "N", "E"],

    # User-defined file naming pattern
    # Example for one common generic case:
    # "{station}_{component}_{date}.sac"
    #
    # Example for your local private dataset (do not publish this in repo):
    # "8O.{station}..{component}.D.{date}.000000.000000_unitmps.sac"
    file_pattern="{station}_{component}_{date}.sac",

    # How to extract dates from filenames:
    # - "underscore_last" assumes last underscore-separated token is date
    # - "dot_index_5" assumes date is token 5 after splitting on "."
    date_extractor="underscore_last",

    window_length=3600,
    step=60,
    zoom_factor=0.05,
    freq_min=0.5,
    freq_max=25.0,
    target_sampling_rate=125,
    save_figures=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_component_roles(components: List[str]) -> Tuple[str, str, str]:
    """
    Infer vertical / horizontal component names from the component list.
    Expected examples:
    - DPZ, DPN, DPE
    - Z, N, E
    """
    vertical = next((c for c in components if c.endswith("Z")), None)
    north = next((c for c in components if c.endswith("N")), None)
    east = next((c for c in components if c.endswith("E")), None)

    if vertical is None or north is None or east is None:
        raise ValueError(
            f"Could not infer component roles from {components}. "
            "Expected one component ending in Z, one in N, and one in E."
        )

    return vertical, north, east


def extract_dates(files: List[str], station: str, mode: str = "underscore_last") -> List[str]:
    """
    Extract unique dates from filenames.

    Supported modes:
    - underscore_last: last underscore-separated token before .sac is date
      Example: STATION_COMPONENT_20200101.sac
    - dot_index_5: token at index 5 after splitting on "."
      Example: 8O.F0208..DPZ.D.20201207.000000.000000_unitmps.sac
    """
    dates = set()

    for f in files:
        name = os.path.basename(f)

        if station not in name:
            continue

        if mode == "underscore_last":
            stem = name.replace(".sac", "")
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            dates.add(parts[-1])

        elif mode == "dot_index_5":
            parts = name.split(".")
            if len(parts) > 5:
                dates.add(parts[5])

        else:
            raise ValueError(
                f"Unsupported date_extractor mode: {mode}. "
                "Use 'underscore_last' or 'dot_index_5'."
            )

    return sorted(dates)


def compute_window_std(
    traces: Dict[str, object],
    component_z: str,
    component_n: str,
    component_e: str,
    window_length: int,
    step: int,
) -> Tuple[List[pd.Timestamp], List[float], List[float], List[float], List[float]]:
    starttime = traces[component_z].stats.starttime
    endtime = traces[component_z].stats.endtime

    time_list: List[pd.Timestamp] = []
    std_z_list: List[float] = []
    std_n_list: List[float] = []
    std_e_list: List[float] = []
    std_sum_list: List[float] = []

    current_start = starttime

    # Only full windows contained within the 24-hour record are used
    while current_start + window_length <= endtime:
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


def select_quiet_window(
    time_list: List[pd.Timestamp],
    std_sum_list: List[float],
    window_length: int,
    step: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, float, int, int]:
    if not time_list:
        raise ValueError("No valid windows were computed.")

    num_steps = int(window_length / step)
    rolling_sums = pd.Series(std_sum_list).rolling(window=num_steps).sum().to_numpy()

    min_index = int(np.nanargmin(rolling_sums))
    if min_index < num_steps - 1:
        raise ValueError("Quiet window selection failed because minimum rolling window is incomplete.")

    start_index = min_index - num_steps + 1
    best_start = pd.to_datetime(time_list[start_index])
    best_end = best_start + pd.Timedelta(seconds=window_length)
    quiet_std_sum = float(rolling_sums[min_index])

    return best_start, best_end, quiet_std_sum, start_index, min_index


def compute_full_day_std(
    traces: Dict[str, object],
    component_z: str,
    component_n: str,
    component_e: str,
) -> float:
    return float(
        np.std(traces[component_z].data) +
        np.std(traces[component_n].data) +
        np.std(traces[component_e].data)
    )


def save_std_csv(
    output_dir: str,
    station: str,
    date_str: str,
    time_list: List[pd.Timestamp],
    std_z_list: List[float],
    std_n_list: List[float],
    std_e_list: List[float],
    std_sum_list: List[float],
    start_index: int,
    end_index: int,
) -> None:
    df = pd.DataFrame({
        "Time_UTC": time_list,
        "STD_Z": std_z_list,
        "STD_N": std_n_list,
        "STD_E": std_e_list,
        "STD_SUM": std_sum_list,
        "Best_Window": [start_index <= i <= end_index for i in range(len(time_list))],
    })
    out_csv = os.path.join(output_dir, f"{station}_{date_str}_std.csv")
    df.to_csv(out_csv, index=False)


def create_qc_plot(
    output_dir: str,
    station: str,
    date_str: str,
    traces: Dict[str, object],
    time_list: List[pd.Timestamp],
    std_z_list: List[float],
    std_n_list: List[float],
    std_e_list: List[float],
    std_sum_list: List[float],
    best_start: pd.Timestamp,
    config: Config,
    component_z: str,
    component_n: str,
    component_e: str,
) -> None:
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    starttime = traces[component_z].stats.starttime
    quiet_start_hr = (UTCDateTime(best_start) - starttime) / 3600
    quiet_end_hr = quiet_start_hr + config.window_length / 3600

    ordered_components = [component_z, component_n, component_e]

    timeseries = {}
    specs = {}
    vmin_global, vmax_global = None, None
    max_ampl = 0.0

    for comp in ordered_components:
        trace = traces[comp].copy()

        factor = int(trace.stats.sampling_rate / config.target_sampling_rate)
        if factor > 1:
            trace.decimate(factor, no_filter=True)

        data = trace.data.astype(np.float32)
        max_ampl = max(max_ampl, float(np.max(np.abs(data))))
        t = trace.times() / 3600

        f, t_spec, sxx = spectrogram(
            data,
            fs=trace.stats.sampling_rate,
            nperseg=512,
            noverlap=256,
        )
        sxx_db = 10 * np.log10(sxx + 1e-20)

        vmin = np.percentile(sxx_db, 10)
        vmax = np.percentile(sxx_db, 98)
        vmin_global = vmin if vmin_global is None else min(vmin_global, vmin)
        vmax_global = vmax if vmax_global is None else max(vmax_global, vmax)

        timeseries[comp] = (t, data)
        specs[comp] = (f, t_spec / 3600, sxx_db)

    fig, axes = plt.subplots(7, 1, figsize=(18, 14), sharex=True)

    std_times = [(UTCDateTime(t) - starttime) / 3600 for t in time_list]
    axes[0].plot(std_times, std_z_list, label=component_z, color="blue")
    axes[0].plot(std_times, std_n_list, label=component_n, color="green")
    axes[0].plot(std_times, std_e_list, label=component_e, color="orange")
    axes[0].plot(std_times, std_sum_list, label="Total STD", linestyle="--", color="black")
    axes[0].axvspan(quiet_start_hr, quiet_end_hr, color="red", alpha=0.3, label="Quiet Window")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    axes[0].set_ylabel("STD")
    axes[0].set_title("Rolling standard deviation (3-component)")
    axes[0].grid(alpha=0.3)

    for i, comp in enumerate(ordered_components):
        t, d = timeseries[comp]
        axes[i + 1].plot(t, d, color="black", linewidth=0.5)
        axes[i + 1].set_ylim(-max_ampl * config.zoom_factor, max_ampl * config.zoom_factor)
        axes[i + 1].axvline(quiet_start_hr, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        axes[i + 1].axvline(quiet_end_hr, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        axes[i + 1].set_ylabel("Amplitude")
        axes[i + 1].set_title(f"{comp} time series")

    for i, comp in enumerate(ordered_components):
        f, t, sxx_db = specs[comp]
        axes[i + 4].imshow(
            sxx_db,
            aspect="auto",
            extent=[t[0], t[-1], f[0], f[-1]],
            origin="lower",
            cmap="viridis",
            vmin=vmin_global,
            vmax=vmax_global,
        )
        axes[i + 4].axvline(quiet_start_hr, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
        axes[i + 4].axvline(quiet_end_hr, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
        axes[i + 4].set_ylim(config.freq_min, config.freq_max)
        axes[i + 4].set_ylabel("Freq [Hz]")
        axes[i + 4].set_title(f"{comp} spectrogram")

    axes[-1].set_xlabel("Time [hours]")
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 1))
        ax.tick_params(axis="x", labelrotation=0)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"combined_{station}_{date_str}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    os.makedirs(CONFIG.base_output_dir, exist_ok=True)

    component_z, component_n, component_e = get_component_roles(CONFIG.components)

    for station in CONFIG.stations:
        for month in CONFIG.months:
            print(f"\nProcessing {station} - {month}")

            data_dir = os.path.join(CONFIG.base_data_dir, month)
            output_dir = os.path.join(CONFIG.base_output_dir, f"{station}_{month}")
            os.makedirs(output_dir, exist_ok=True)

            files = glob.glob(os.path.join(data_dir, "*.sac"))
            dates = extract_dates(files, station, mode=CONFIG.date_extractor)

            comparison_data = []
            quiet_std_data = []

            for date in dates:
                try:
                    traces = {}

                    for comp in CONFIG.components:
                        pattern = CONFIG.file_pattern.format(
                            station=station,
                            component=comp,
                            date=date,
                        )
                        file_path = os.path.join(data_dir, pattern)

                        if not os.path.exists(file_path):
                            raise FileNotFoundError(file_path)

                        traces[comp] = read(file_path)[0]

                    time_list, std_z_list, std_n_list, std_e_list, std_sum_list = compute_window_std(
                        traces=traces,
                        component_z=component_z,
                        component_n=component_n,
                        component_e=component_e,
                        window_length=CONFIG.window_length,
                        step=CONFIG.step,
                    )

                    best_start, best_end, quiet_std_sum, start_index, end_index = select_quiet_window(
                        time_list=time_list,
                        std_sum_list=std_sum_list,
                        window_length=CONFIG.window_length,
                        step=CONFIG.step,
                    )

                    std_full_sum = compute_full_day_std(
                        traces=traces,
                        component_z=component_z,
                        component_n=component_n,
                        component_e=component_e,
                    )

                    comparison_data.append({
                        "Station": station,
                        "Date": date,
                        "Quiet_STD_SUM": quiet_std_sum,
                        "FullDay_STD_SUM": std_full_sum,
                        "Ratio_Quiet_to_Full": quiet_std_sum / std_full_sum if std_full_sum != 0 else np.nan,
                        "Quiet_Window_Start": best_start,
                        "Quiet_Window_End": best_end,
                    })

                    quiet_std_data.append({
                        "Station": station,
                        "Date": date,
                        "Quiet_STD_SUM": quiet_std_sum,
                        "Quiet_Window_Start": best_start,
                        "Quiet_Window_End": best_end,
                    })

                    save_std_csv(
                        output_dir=output_dir,
                        station=station,
                        date_str=date,
                        time_list=time_list,
                        std_z_list=std_z_list,
                        std_n_list=std_n_list,
                        std_e_list=std_e_list,
                        std_sum_list=std_sum_list,
                        start_index=start_index,
                        end_index=end_index,
                    )

                    if CONFIG.save_figures:
                        create_qc_plot(
                            output_dir=output_dir,
                            station=station,
                            date_str=date,
                            traces=traces,
                            time_list=time_list,
                            std_z_list=std_z_list,
                            std_n_list=std_n_list,
                            std_e_list=std_e_list,
                            std_sum_list=std_sum_list,
                            best_start=best_start,
                            config=CONFIG,
                            component_z=component_z,
                            component_n=component_n,
                            component_e=component_e,
                        )

                    print(f"{date} → Quiet window: {best_start} to {best_end}")

                except Exception as e:
                    print(f"Error processing {station} {date}: {e}")
                    traceback.print_exc()

            # Save summaries inside the station/month folder
            summary_csv = os.path.join(output_dir, f"{station}_{month}_quiet_vs_full_summary.csv")
            quiet_csv = os.path.join(output_dir, f"{station}_{month}_quiet_std_summary.csv")

            pd.DataFrame(comparison_data).to_csv(summary_csv, index=False)
            pd.DataFrame(quiet_std_data).to_csv(quiet_csv, index=False)

            print(f"Saved summary files for {station} - {month}")


if __name__ == "__main__":
    main()