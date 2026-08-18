# NHLBI_144 Analysis Pipeline

This repository contains standalone Python scripts for processing calcium-imaging videos, preparing atlas-alignment inputs, generating ROI trend curves, detecting peak events, detecting behavior-motion periods, and summarizing calcium activity during motion vs. stationary periods.

The repository is script-driven rather than package-based.

## Repository Contents

- `getFirstFrame4Atlas.py`: batch extract the first frame from each `.mp4` for atlas alignment
- `1_pipeline_v5.py`: preprocess raw calcium videos and write `final.nrrd`
- `2_trendChart_v3.py`: generate ROI trend curves, ROI totals, gray masks, and per-ROI plots from `final.nrrd`
- `3_peakAnalysis_v2.py`: detect peak events from ROI curve CSV files and compile results across sessions
- `4_behavior_time_signature_batch_v4.py`: detect motion events from behavior videos and export motion timestamps
- `5_motion_stationary_total_v2.py`: compute ROI calcium totals during motion vs. stationary periods
- `results/315_sub_v2.txt`: example ROI color-to-label definition file
- `results/315_sub_color.png`: example original unaligned atlas image
- `results/315_sub_color_aligned_example.png`: example aligned atlas image

## Recommended Environment

Use Python 3.9+ in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn SimpleITK opencv-python matplotlib pillow
```

### Conda Environment Setup

To create a new conda environment for this project:

```bash
conda create -n nhlbi144 python=3.9 -y
conda activate nhlbi144
python -m pip install numpy pandas scipy scikit-learn SimpleITK opencv-python matplotlib pillow
```

## Expected Data Layout

The scripts assume a dataset layout like this:

```text
DATA_ROOT/
  Animal_ID/
    Day_1/
      video1 SS.mp4
      video1 LH.mp4
      Behavior/
        video1 SS behavior.mp4
```

`1_pipeline_v5.py` and `getFirstFrame4Atlas.py` scan:

```text
<data_root>/*/*/*.mp4
```

`1_pipeline_v5.py` then filters to files whose path contains `SS`.

The processed output layout is:

```text
OUTPUT_ROOT/
  Animal_ID/
    Day_1/
      video1 SS/
        final.nrrd
        315_sub_color_aligned.png
```

For `2_trendChart_v3.py` and `5_motion_stationary_total_v2.py`, each processed session folder must contain:

- `final.nrrd`
- `315_sub_color_aligned.png`

## Atlas Alignment Assets

The `results/` folder in this repository contains examples/reference files only:

- `results/315_sub_v2.txt`: example ROI index, color, and abbreviation table
- `results/315_sub_color.png`: example original unaligned atlas
- `results/315_sub_color_aligned_example.png`: example aligned atlas for reference

The analysis scripts are still written to use a separate external `results` folder, not this repository copy.

In particular, `2_trendChart_v3.py` and `5_motion_stationary_total_v2.py` expect the ROI definition file at:

```text
../results/315_sub_v2.txt
```

relative to the working directory from which the script is run.

Use `getFirstFrame4Atlas.py` to extract the first frame from each video, align the atlas to that frame outside this repo, and save the aligned result into each processed session folder as:

```text
315_sub_color_aligned.png
```

## Pipeline Order

The intended workflow is:

1. Run `getFirstFrame4Atlas.py` to extract first-frame images for atlas alignment
2. Align the atlas to each extracted first frame and create `315_sub_color_aligned.png`
3. Run `1_pipeline_v5.py` on raw calcium videos to generate `final.nrrd`
4. Copy or save each aligned atlas into the matching processed session folder
5. Run `2_trendChart_v3.py` to generate ROI curves and summary tables
6. Run `3_peakAnalysis_v2.py` on selected ROI curve CSV files
7. Run `4_behavior_time_signature_batch_v4.py` on the raw dataset to generate motion timestamps
8. Run `5_motion_stationary_total_v2.py` to split calcium activity into motion and stationary periods

## How To Run

### 1. Extract first frames for atlas alignment

This script reads the first frame from every `.mp4`, downsizes it by half to match the main processing pipeline, and writes a `.jpg` into a mirrored output folder structure.

```bash
python getFirstFrame4Atlas.py \
  --data_root /path/to/raw_dataset \
  --output_root /path/to/atlas_frames
```

Example output:

```text
/path/to/atlas_frames/Animal_ID/Day_1/video1 SS/video1 SS-0.jpg
```

These extracted images are intended to be used as the background/reference for atlas alignment.

### 2. Preprocess calcium videos

This script reads `.mp4` files, converts frames to grayscale, downsamples spatially by 2, computes dF/F, performs global randomized SVD, detrending, heartbeat filtering, high-pass filtering, and writes `final.nrrd`.

```bash
python 1_pipeline_v5.py \
  --data_root /path/to/raw_dataset \
  --output_root /path/to/processed_output
```

Optional:

```bash
python 1_pipeline_v5.py \
  --data_root /path/to/raw_dataset \
  --output_root /path/to/processed_output \
  --save_steps
```

Main outputs per processed session:

- `final.nrrd`
- `dfof_ch1.nrrd` if `--save_steps` is enabled

### 3. Add aligned atlas masks

For each processed session folder, place the aligned atlas mask in the same folder as `final.nrrd` and name it exactly:

```text
315_sub_color_aligned.png
```

This file is required by:

- `2_trendChart_v3.py`
- `5_motion_stationary_total_v2.py`

### 4. Generate ROI trend curves and per-session summaries

This script reads `final.nrrd`, applies the aligned color mask, computes ROI-wise trend curves, saves plots, and writes a normalized total activity table. Use average_factor=1 for full resolution analysis.

```bash
python 2_trendChart_v3.py \
  --root_folder /path/to/processed_output \
  --average_factor 1
```

Optional prefix:

```bash
python 2_trendChart_v3.py \
  --root_folder /path/to/processed_output \
  --average_factor 1 \
  --prefix run1
```

Outputs written into each session folder include:

- `gray_mask.png`
- `total.csv`
- `final_curve_<ROI_LABEL>.csv`
- `delta_<ROI_ID>_<ROI_LABEL>.png`

### 5. Compile peak-event results from ROI curves

This script looks for one ROI curve CSV name in every session folder and compiles detected events into one CSV.

```bash
python 3_peakAnalysis_v2.py \
  --root_folder /path/to/processed_output \
  --key final_curve_SS \
  --fps 1.0 \
  --save_plots
```

Notes:

- `--key` is the CSV base name without `.csv`
- `--fps` is the target resampling rate for event detection, not the raw recording rate
- output is written to the root folder as `compiled_peak_events_<key>_resampled_at_<fps>Hz.csv`

### 6. Detect motion periods from behavior videos

This script scans the raw dataset root, looks for behavior videos under `Behavior/`, detects motion events, optionally saves plots, and exports a CSV of motion periods.

```bash
python 4_behavior_time_signature_batch_v4.py \
  /path/to/raw_dataset \
  --output_csv motion_event_timestamps.csv \
  --plot_dir motion_plots
```

Optional parameters:

- `--save_frames`
- `--max_time`
- `--diff_thresh`
- `--min_event_frames`
- `--merge_gap`

Expected behavior-video naming:

```text
<day_folder>/<base_video>.mp4
<day_folder>/Behavior/<base_video> behavior.mp4
```

Output columns include:

- `animal_id`
- `day_id`
- `video_name`
- `event_number`
- `start_frame`
- `end_frame`
- `start_time_sec`
- `end_time_sec`
- `duration_sec`

### 7. Summarize calcium activity during motion vs. stationary periods

This script combines processed calcium data with the motion timestamp CSV from step 6.

```bash
python 5_motion_stationary_total_v2.py \
  --root_folder /path/to/processed_output \
  --motion_data /path/to/motion_event_timestamps.csv \
  --fps 30
```

Optional prefix:

```bash
python 5_motion_stationary_total_v2.py \
  --root_folder /path/to/processed_output \
  --motion_data /path/to/motion_event_timestamps.csv \
  --fps 30 \
  --prefix motionsplit
```

Outputs written into each session folder include:

- `gray_mask.png`
- `total.csv` with `motion` and `stationary` columns

## Script Inputs And Outputs Summary

| Step | Script | Main Input | Main Output |
| --- | --- | --- | --- |
| 1 | `getFirstFrame4Atlas.py` | raw `.mp4` files | first-frame `.jpg` files for atlas alignment |
| 2 | `1_pipeline_v5.py` | raw calcium `.mp4` files | `final.nrrd` |
| 3 | manual atlas alignment | first-frame `.jpg` + atlas assets | `315_sub_color_aligned.png` |
| 4 | `2_trendChart_v3.py` | `final.nrrd` + aligned atlas mask | ROI curve CSVs, ROI plots, `total.csv` |
| 5 | `3_peakAnalysis_v2.py` | ROI curve CSVs | compiled peak-event CSV |
| 6 | `4_behavior_time_signature_batch_v4.py` | behavior `.mp4` files | motion timestamp CSV |
| 7 | `5_motion_stationary_total_v2.py` | `final.nrrd` + motion CSV + aligned atlas mask | motion/stationary ROI totals |

## Current Repository Notes

- `2_trendChart_v3.py` no longer imports `helpers`.
- The `results/` folder in this repository is for examples/reference only.
- `2_trendChart_v3.py` and `5_motion_stationary_total_v2.py` still reference `../results/315_sub_v2.txt`, so they currently expect a separate external `results` folder that matches that hard-coded relative path.
- `1_pipeline_v5.py` only processes files whose path contains `SS`.
- `getFirstFrame4Atlas.py` processes all `.mp4` files under the dataset root.

## Practical Notes For New Team Members

- Run commands from the repository root unless you intentionally change the hard-coded paths in the scripts.
- Verify one session end to end before launching the full dataset.
- Before running `2_trendChart_v3.py` or `5_motion_stationary_total_v2.py`, confirm both `final.nrrd` and `315_sub_color_aligned.png` are present in each processed session folder.
- Confirm the external `../results/315_sub_v2.txt` path resolves correctly in your runtime environment.
- Use `results/315_sub_color_aligned_example.png` as a visual reference for how the aligned atlas should look.
