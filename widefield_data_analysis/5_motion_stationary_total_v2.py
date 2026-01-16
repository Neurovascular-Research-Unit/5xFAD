"""
Dataset Processor to generate fully normalized motion/stationary activity tables.

This script processes a dataset organized in the structure:
  root_folder/animal_folder/video_folder/

The script uses an external CSV file containing motion timestamps to separate
calcium activity into 'motion' and 'stationary' periods.

It normalizes the total activity for each category by both the
area of the Region of Interest (ROI) and the number of frames in that
category (length). This produces a standardized metric of 
"average activity change per pixel per frame", allowing for direct comparison
across different animals and recording sessions.

Usage:
  python motion_analysis_final.py --root_folder /path/to/your/dataset --motion_data /path/to/motion.csv [--fps 30] [--prefix OUTPUT_PREFIX]

"""
import os
import argparse
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk
import pandas as pd

from PIL import ImageColor

FRAME_LIMIT = 53998
COLOR_INDEX = '../results/315_sub_v2.txt'
COLOR_MASK = '315_sub_color_aligned.png'

def loadColorRampAndLabel(fName):
  colorDict = {}
  colorLabelDict = {}
  with open(fName) as f:
    colors = [x.rstrip() for x in f]
    for cc in colors:
      tokens = cc.split(' ')
      colorDict[int(tokens[0])] = ImageColor.getcolor(tokens[1], "RGB")
      colorLabelDict[int(tokens[0])] = tokens[2]
  print('Color Ramp', colorDict)
  print('Color Label', colorLabelDict)
  return colorDict, colorLabelDict

def colorMask2GrayMask(c_msk, color_ramp):
  g_msk = np.zeros((c_msk.shape[0], c_msk.shape[1]), dtype=np.uint16)
  pixel_areas = {}
  for idx in color_ramp:
    color = color_ramp[idx]
    indices = np.where(np.all(c_msk == color, axis=-1))
    g_msk[indices[0], indices[1]] = idx
    pixel_areas[idx] = len(indices[0])
  return g_msk, pixel_areas

def getMotionStationaryCalcium(vol, msk, c_ramp, areas, folder_path, motion_df, fps):
    """
    Calculates fully normalized calcium activity, separated into motion and stationary periods.
    The normalization formula is: Total_Signal / (Area_of_ROI * Number_of_Frames)
    This produces a metric of "average activity change per pixel per frame".
    """
    start_time = time.time()
    folder_name = Path(folder_path).name
    target_video_name = f"{Path(folder_name).stem} behavior.mp4"
    print(f"  Mapping folder '{folder_name}' to motion video name: '{target_video_name}'")

    video_motion_periods = motion_df[motion_df['video_name'] == target_video_name]
    if video_motion_periods.empty:
        print(f"    WARNING: No motion data found for '{target_video_name}'. All frames will be considered stationary.")

    # --- Pre-computation ---
    total_frames = min(vol.shape[0], FRAME_LIMIT)
    is_motion_frame = np.zeros(total_frames, dtype=bool)

    if not video_motion_periods.empty:
        for _, row in video_motion_periods.iterrows():
            start_frame = int(row['start_time_sec'] * fps)
            end_frame = int(row['end_time_sec'] * fps)
            valid_end_frame = min(end_frame, total_frames)
            if start_frame < valid_end_frame:
                is_motion_frame[start_frame:valid_end_frame] = True

    # --- NEW: Calculate the number of frames in each category for normalization ---
    num_motion_frames = np.sum(is_motion_frame)
    num_stationary_frames = total_frames - num_motion_frames
    print(f"    Normalizing by {num_motion_frames} motion frames and {num_stationary_frames} stationary frames.")

    # Flatten the mask ONCE for use with bincount
    flat_mask = msk.ravel()
    
    max_roi_id = max(c_ramp.keys()) if c_ramp else 0
    motion_sums = np.zeros(max_roi_id + 1, dtype=np.float64)
    stationary_sums = np.zeros(max_roi_id + 1, dtype=np.float64)
    
    print(f"    Processing {total_frames} frames...")

    # --- Main Processing Loop: Iterate through the volume ONCE ---
    for i in range(total_frames):
        flat_frame = np.abs(vol[i]).ravel()
        frame_roi_sums = np.bincount(flat_mask, weights=flat_frame, minlength=max_roi_id + 1)
        if is_motion_frame[i]:
            motion_sums += frame_roi_sums
        else:
            stationary_sums += frame_roi_sums
    
    # --- MODIFIED: Final double normalization (by area and time) ---
    motion_normalized_list = []
    stationary_normalized_list = []
    
    for idx in c_ramp:
        area = areas.get(idx, 0)
        
        # Calculate normalized motion value
        if area > 0 and num_motion_frames > 0:
            # Normalize by both area and the number of motion frames
            norm_motion = motion_sums[idx] / (area * num_motion_frames)
            motion_normalized_list.append(norm_motion)
        else:
            motion_normalized_list.append(0)

        # Calculate normalized stationary value
        if area > 0 and num_stationary_frames > 0:
            # Normalize by both area and the number of stationary frames
            norm_stationary = stationary_sums[idx] / (area * num_stationary_frames)
            stationary_normalized_list.append(norm_stationary)
        else:
            stationary_normalized_list.append(0)
            
    end_time = time.time()
    print(f"    [OPTIMIZED] Calcium calculation took {end_time - start_time:.2f} seconds.")

    return motion_normalized_list, stationary_normalized_list

def process_data_in_folder(folder_path, motion_df, fps, prefix=""):
  volume_path = os.path.join(folder_path, 'final.nrrd')
  vol = sitk.GetArrayFromImage(sitk.ReadImage(volume_path))

  print('Frames:', vol.shape[0], np.amin(vol), np.amax(vol))

  colorMsk_path = os.path.join(folder_path, COLOR_MASK)
  colorMsk = sitk.GetArrayFromImage(sitk.ReadImage(colorMsk_path, sitk.sitkVectorUInt8))[:, :, :3]

  colorRamp, colorLabel = loadColorRampAndLabel(COLOR_INDEX)
  grayMsk, areas = colorMask2GrayMask(colorMsk, colorRamp)

  grayMskName = prefix + '_gray_mask.png' if prefix else 'gray_mask.png'
  sitk.WriteImage(sitk.GetImageFromArray(grayMsk), os.path.join(folder_path, grayMskName))

  motion_calcium, stationary_calcium = getMotionStationaryCalcium(vol, grayMsk, colorRamp, areas, folder_path, motion_df, fps)

  table = pd.DataFrame({
      'Idx': colorRamp.keys(),
      'Abbreviation': list(colorLabel.values()),
      'Areas': list(areas.values()),
      'motion': motion_calcium,
      'stationary': stationary_calcium
  })
  
  totalFile = prefix + '_total.csv' if prefix else 'total.csv'
  table.to_csv(os.path.join(folder_path, totalFile), index=False)
  print(f"  Successfully generated summary table: {totalFile}")
  
  return True

def main(root_folder, motion_data_path, fps, prefix=""):
  try:
    motion_df = pd.read_csv(motion_data_path)
    print(f"Successfully loaded motion data from: {motion_data_path}")
  except FileNotFoundError:
    print(f"Error: Motion data file not found at {motion_data_path}")
    return
  
  total_folders, processed_folders, skipped_folders = 0, 0, 0

  for animal_folder in os.listdir(root_folder):
    animal_path = os.path.join(root_folder, animal_folder)
    if not os.path.isdir(animal_path):
      continue

    print(f"Processing animal: {animal_folder}")

    for video_folder in os.listdir(animal_path):
      video_path = os.path.join(animal_path, video_folder)
      if not os.path.isdir(video_path):
        continue

      total_folders += 1
      png_file = os.path.join(video_path, COLOR_MASK)

      if os.path.exists(png_file):
        print(f"  Found required PNG in: {video_folder}. Processing...")
        if process_data_in_folder(video_path, motion_df, fps, prefix):
          processed_folders += 1
      else:
        print(f"  Skipping {video_folder} - required PNG not found")
        skipped_folders += 1

  print("\nProcessing Summary:")
  print(f"Total folders examined: {total_folders}")
  print(f"Folders processed: {processed_folders}")
  print(f"Folders skipped: {skipped_folders}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Process dataset to calculate fully normalized motion and stationary calcium activity.")
  parser.add_argument("--root_folder", required=True, help="Root folder containing the dataset")
  parser.add_argument("--motion_data", required=True, help="Path to the CSV file containing motion period data")
  parser.add_argument("--fps", type=int, default=30, help="Frames per second of the videos (default: 30)")
  parser.add_argument("--prefix", default="", help="Optional prefix for output filenames")
  args = parser.parse_args()

  print(args, flush=True)

  print(f"Starting to process dataset at: {args.root_folder}")
  if args.prefix:
    print(f"Using prefix for output files: '{args.prefix}'")
  main(args.root_folder, args.motion_data, args.fps, args.prefix)
  print("Processing complete!")
