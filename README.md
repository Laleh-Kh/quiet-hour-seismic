# Quiet Hour Selection

This repository contains a simple Python script for selecting the quietest one-hour window from three-component seismic waveform data.

The current version reads three built-in ObsPy example waveform files, computes the standard deviation for the `Z`, `N`, and `E` components in sliding windows, sums those values to form `std_sum`, and then applies a rolling sum to identify the quiet hour. If next-day data are available, the script uses the first hour of the following day when computing the final 23:00-24:00 window.

## Method

1. Compute `STD_Z`, `STD_N`, and `STD_E` for each candidate window.
2. Compute `STD_SUM = STD_Z + STD_N + STD_E`.
3. Compute a rolling sum of `STD_SUM`.
4. Select the quiet hour from the minimum of that rolling-sum curve.

## Plot

The output figure has two panels:

- top: `Z`, `N`, `E`, and `Total STD`
- bottom: the rolling-sum selection metric

The selected quiet hour is highlighted on the lower panel because that is the metric used for the final selection.

## Requirements

- Python
- `numpy`
- `pandas`
- `matplotlib`
- `obspy`

## Run

```bash
python select-quiet-hour.py
```

## Output

Running the script creates a `quiet_hour_output` folder next to the script and writes:

- `quiet_hour_selection.csv`
- `quiet_hour_metrics.csv`
- `quiet_hour_metrics.png`
- `quiet_hour_metrics.pdf`
