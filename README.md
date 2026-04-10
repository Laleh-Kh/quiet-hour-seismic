# quiet-hour-seismic

Select the quietest time window in continuous passive seismic data using a sliding standard deviation (STD) method.

## Overview

This script identifies the most stable (lowest-noise) time window within a continuous 24-hour seismic record.

A fixed-length window (default: 1 hour) is moved across the data, and the standard deviation of the three components is computed. The window with the minimum combined STD is selected as the quietest interval. Only windows fully contained within each record are evaluated, so up to one window length at the end may be excluded.

This workflow was developed for preprocessing ambient seismic data (e.g., HVSR), but is general and can be applied to any continuous seismic dataset.

## Method

- Slide a fixed window across the record  
- Compute STD for each component  
- Combine into a single noise metric  
- Select the window with minimum noise  

## Usage

Edit the `CONFIG` block in `select_quiet_hour.py`:

```python
CONFIG = Config(
    base_data_dir="PATH_TO_DATA",
    base_output_dir="PATH_TO_OUTPUT",
    stations=["STATION_NAME"],
    months=["MONTH_NAME"],
    components=["Z", "N", "E"],
    file_pattern="{station}_{component}_{date}.sac",
)
```
  
# Run
   `python select_quiet_hour.py`
## Inputs
- Continuous 3-component passive seismic data
- One file per component per day
- User-defined file naming pattern
## Outputs
- Per-window STD CSV files
- Summary CSV files (quiet window start/end times)
- QC plots (STD, time series, spectrograms)
## Notes
- Components must end with Z, N, and E
- Default window length is 1 hour
- Partial windows at the end of each day are excluded
