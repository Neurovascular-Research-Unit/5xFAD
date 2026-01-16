import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The frame rate of the raw input data is fixed at 30 Hz.
RAW_INPUT_FPS = 30.0

def find_peak_events(curve, baseline):
    """
    Finds continuous periods (events) where the curve is above a baseline.

    An event starts when the signal crosses above the baseline and ends when
    it crosses back below.

    Args:
        curve (np.array): The 1D time series data.
        baseline (float): The baseline value.

    Returns:
        list of dicts: A list where each dictionary represents a found event
                      and contains its start, end, duration, and peak info.
    """
    # Create a boolean array, True where curve is above baseline
    # per Amreen, points where the curve touches baseline
    # should be considered as event boundaries
    above_baseline = curve > baseline
    
    # Find the indices where the state changes
    # A +1 indicates a rising edge (False -> True)
    # A -1 indicates a falling edge (True -> False)
    diff = np.diff(above_baseline.astype(int))
    
    # Get the start and end indices of events
    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0]
    
    # --- Handle edge cases ---
    # 1. The curve starts already above the baseline
    if above_baseline[0]:
        start_indices = np.insert(start_indices, 0, 0)
        
    # 2. The curve ends while still above the baseline
    if above_baseline[-1]:
        end_indices = np.append(end_indices, len(curve) - 1)
        
    # --- Sanity check ---
    # Ensure every start has a corresponding end
    if len(start_indices) != len(end_indices):
        min_len = min(len(start_indices), len(end_indices))
        start_indices = start_indices[:min_len]
        end_indices = end_indices[:min_len]

    # --- Characterize each event ---
    events = []
    for start, end in zip(start_indices, end_indices):
        if end < start:
            continue
        
        event_slice = curve[start:end + 1]
        duration = len(event_slice)
        
        # Find the peak within this event
        peak_local_idx = np.argmax(event_slice)
        peak_global_idx = start + peak_local_idx
        peak_height = event_slice[peak_local_idx]
        
        events.append({
            "start_index": start,
            "end_index": end,
            "duration_samples": duration,
            "peak_index": peak_global_idx,
            "peak_height": peak_height,
        })
        
    return events

def analyze_curve_file(file_path, resample_fps=30.0, save_plots=False):
    """
    Reads a curve, resamples it by binning, analyzes peak events, and returns results.

    Args:
        file_path (str): Path to the input CSV file.
        resample_fps (float): The target FPS for resampling. A lower value
                              results in a larger time window for binning.
        save_plots (bool): If True, a visualization plot is saved.
    """
    print(f"\n--- Analyzing file: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        raw_curve = df.iloc[:, 1].to_numpy()
        if len(raw_curve) < 2:
            print("  Warning: Curve has less than 2 data points. Skipping.")
            return None
    except Exception as e:
        print(f"  Error reading or processing file: {e}. Skipping.")
        return None

    # --- Resampling Step ---
    # Calculate the number of original samples to average for each new sample.
    if resample_fps <= 0:
        print(f"  Error: resample_fps must be positive. Skipping.")
        return None
    
    window_size_samples = int(round(RAW_INPUT_FPS / resample_fps))

    if window_size_samples <= 1:
        # No resampling needed, use the raw curve
        print("  No resampling applied (window size <= 1).")
        resampled_curve = raw_curve
        window_size_samples = 1
    else:
        window_duration_s = window_size_samples / RAW_INPUT_FPS
        print(f"  Resampling with a moving window of {window_size_samples} samples ({window_duration_s:.2f} seconds).")
        # Truncate the curve to be perfectly divisible by the window size
        num_windows = len(raw_curve) // window_size_samples
        if num_windows == 0:
            print("  Warning: Curve is shorter than one window size. Skipping.")
            return None
        truncated_len = num_windows * window_size_samples
        binnable_curve = raw_curve[:truncated_len]
        # Reshape into bins and calculate the mean of each bin
        resampled_curve = binnable_curve.reshape(num_windows, window_size_samples).mean(axis=1)

    # Define baseline as 20th percentile of the *resampled* curve's data range
    predefined_baseline = np.percentile(resampled_curve, 20)
    print(f"  Predefined Baseline (from resampled data): {predefined_baseline:.3f}")

    # Find all peak events on the resampled curve
    events_resampled = find_peak_events(resampled_curve, predefined_baseline)
    
    if not events_resampled:
        print("  No peak events found above the baseline.")
        return None
    
    print(f"  Found {len(events_resampled)} peak events.")

    # --- Map results back to original resolution ---
    events_original_resolution = []
    for event in events_resampled:
        # Convert start/end indices from resampled coordinates to original coordinates
        start_orig = event["start_index"] * window_size_samples
        end_orig = (event["end_index"] + 1) * window_size_samples - 1
        
        # Ensure end_orig does not exceed the bounds of the raw curve
        end_orig = min(end_orig, len(raw_curve) - 1)

        # Find the true peak within the corresponding slice of the *raw* data
        event_slice_orig = raw_curve[start_orig : end_orig + 1]
        peak_local_idx_orig = np.argmax(event_slice_orig)
        peak_global_idx_orig = start_orig + peak_local_idx_orig
        peak_height_orig = event_slice_orig[peak_local_idx_orig]
        
        events_original_resolution.append({
            "start_index": start_orig,
            "end_index": end_orig,
            "duration_samples": end_orig - start_orig + 1,
            "peak_index": peak_global_idx_orig,
            "peak_height": peak_height_orig,
        })
        
    results_df = pd.DataFrame(events_original_resolution)
    
    # Add time-based calculations using the original raw FPS
    time_per_sample = 1.0 / RAW_INPUT_FPS
    results_df["duration_s"] = results_df["duration_samples"] * time_per_sample
    results_df["peak_time_s"] = results_df["peak_index"] * time_per_sample
        
    if save_plots:
        base_name = Path(file_path).stem
        output_plot_path = Path(file_path).parent / f"{base_name}_events_resampled_at_{resample_fps}Hz.png"
        
        plt.figure(figsize=(15, 7))
        # Plot original raw data
        plt.plot(raw_curve, label='Raw Signal (30 FPS)', zorder=1, color='gray', alpha=0.7)
        
        # Create x-axis for the resampled curve to plot it correctly
        if window_size_samples > 1:
            x_resampled = np.arange(len(resampled_curve)) * window_size_samples + (window_size_samples / 2)
            plt.plot(x_resampled, resampled_curve, label=f'Binned Signal ({resample_fps} Hz)', zorder=2, color='black', alpha=0.9)

        # Plot baseline and peaks
        plt.hlines(predefined_baseline, 0, len(raw_curve), color='green', linestyle='--', label=f'Baseline ({predefined_baseline:.3f})', zorder=3)
        plt.plot(results_df["peak_index"], results_df["peak_height"], "x", color='red', markersize=8, label='Peak within Event (on Raw)', zorder=4)
        
        # Highlight event durations on the raw curve
        for i, event in results_df.iterrows():
            start_idx = int(event["start_index"])
            end_idx = int(event["end_index"])

            plt.fill_between(np.arange(start_idx, end_idx + 1), 
                             predefined_baseline, 
                             raw_curve[start_idx:end_idx + 1], 
                             color='orange',
                             alpha=0.5, label='Peak Events' if i==0 else "", zorder=0)

        plt.legend()
        plt.xlabel("Sample Index (at 30 FPS)")
        plt.ylabel("Amplitude")
        plt.title(f"Peak Event Analysis for\n{Path(file_path).name}")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        plt.savefig(output_plot_path)
        plt.close()
        print(f"  Saved visualization to: {output_plot_path}")
        
    return results_df

def main():
    parser = argparse.ArgumentParser(
        description="Analyze peak events in time series curves by resampling the data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root_folder", required=True, help="Root folder containing animal_id/session_id subdirectories.")
    parser.add_argument("--key", required=True, help="The base name of the CSV file to analyze (e.g., 'final_curve_SS').")
    parser.add_argument("--fps", type=float, default=30.0, 
                        help="Target FPS for resampling the raw 30Hz data. \n"
                             "This defines the time window for binning before event detection.\n"
                             "Example: --fps 1.0  -> 1 second window (30 samples).\n"
                             "Example: --fps 0.2  -> 5 second window (150 samples).\n"
                             "Example: --fps 0.05 -> 20 second window (600 samples).\n"
                             "Default is 30.0, which means no resampling.")
    parser.add_argument("--save_plots", action="store_true", help="If set, save a visualization plot for each analyzed curve.")
    
    args = parser.parse_args()
    
    print("Starting Peak Event Analysis...")
    print(f"Root Folder: {args.root_folder}")
    print(f"Target Key: {args.key}")
    
    all_results = []

    for animal_folder in os.listdir(args.root_folder):
        animal_path = os.path.join(args.root_folder, animal_folder)
        if not os.path.isdir(animal_path): continue
            
        for session_folder in os.listdir(animal_path):
            session_path = os.path.join(animal_path, session_folder)
            if not os.path.isdir(session_path): continue
                
            csv_filename = f"{args.key}.csv"
            target_file_path = os.path.join(session_path, csv_filename)
            
            if os.path.exists(target_file_path):
                results_df = analyze_curve_file(target_file_path, args.fps, args.save_plots)
                
                if results_df is not None and not results_df.empty:
                    results_df.insert(0, 'Animal_ID', animal_folder)
                    results_df.insert(1, 'Session_ID', session_folder)
                    all_results.append(results_df)

    if not all_results:
        print("\nAnalysis complete. No peak events were found to compile.")
        return
        
    compiled_df = pd.concat(all_results, ignore_index=True)
    
    # Construct the final output file name
    output_filename = f"compiled_peak_events_{args.key}_resampled_at_{args.fps}Hz.csv"
    output_path = os.path.join(args.root_folder, output_filename)
    
    compiled_df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"\nAnalysis complete. Compiled results for {len(all_results)} files saved to: {output_path}")

if __name__ == "__main__":
    main()
