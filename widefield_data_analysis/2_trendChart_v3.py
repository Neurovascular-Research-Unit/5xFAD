"""
Dataset Processor to generate trend graphs, tables, from 'final.nrrd'

This script processes a dataset organized in the structure:
  root_folder/animal_folder/video_folder/

For each video_folder, it checks for the presence of a specific PNG file 
('315_sub_color_aligned.png'). If the file exists, the script processes 
the data in that folder. If not, it skips the folder.

The script writes results directly to the same folder as the source data,
placing output files alongside the original PNG file.

Usage:
  python trendChart_v3.py --root_folder /path/to/your/dataset [--average_factor FACTOR] [--prefix OUTPUT_PREFIX]
"""

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import pandas as pd

from PIL import ImageColor
from scipy.signal import find_peaks

from helpers import *

FRAME_LIMIT = 53998 #frame number cut-off for the new videos
COLOR_INDEX = '../results/315_sub_v2.txt'
COLOR_MASK = '315_sub_color_aligned.png'
SAVE_CURVE_CSV = True # if save curves as csv files

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
    print(idx, color, flush=True)

    # finding indices of all pixels with certain color
    indices = np.where(np.all(c_msk == color, axis=-1))
    g_msk[ indices[0], indices[1] ] = idx

    pixel_areas[idx] = len(indices[0]) # number of pixels
  
  return g_msk, pixel_areas
  
def getTrendCurve(vol, msk, average_factor):
  trend = []
  for frame in vol:
    frame[msk == 0] = 0
    trend.append(np.average(np.abs(frame)))

  if average_factor > 1:
    n = average_factor
    trend_lists = [trend[i:i + n] for i in range(0, len(trend), n)]
    trend = []
    for ll in trend_lists:
      trend.append(np.average(ll))

  return trend

def getTrendCurves(vol, msk, c_ramp, average_factor):
  print(np.unique(msk), flush=True)

  trendList = []
  for idx in c_ramp:
    indices = np.where(msk == idx)

    trend = []
    for frame in vol:  
      pixels = frame[indices[0], indices[1]]
      #print(idx, indices, pixels, vol.shape, msk.shape, frame.shape)
      trend.append(np.average(np.abs(pixels)))

    if average_factor > 1:
      n = average_factor
      trend_lists = [trend[i:i+n] for i in range(0, len(trend), n)]
      newTrend = []
      for ll in trend_lists:
        newTrend.append(np.average(ll))
      trendList.append(newTrend)
    else:
      trendList.append(trend)

  return trendList

def getTotalCalcium(vol, msk, c_ramp, areas):
  newVol = vol[:FRAME_LIMIT]
  calciumList = []
  for idx in c_ramp:
    indices = np.where(msk == idx)

    calcium = 0
    for frame in newVol:
      pixels = frame[indices[0], indices[1]]
      calcium += np.sum(np.abs(pixels))
    
    if areas[idx] == 0:
      calciumList.append(0)
    else:
      calciumList.append(calcium/areas[idx])

  return calciumList
  
def process_data_in_folder(folder_path, average_factor=1, prefix=""):
  volume = os.path.join(folder_path, 'final.nrrd')
  vol = sitk.GetArrayFromImage(sitk.ReadImage(volume))
  sigma = 2
  #vol = removeExtremes(vol, sigma)
  
  print('Frames:', vol.shape[0], np.amin(vol), np.amax(vol))
  
  colorMsk = sitk.ReadImage(os.path.join(folder_path, COLOR_MASK), sitk.sitkVectorUInt8)
  colorMsk = sitk.GetArrayFromImage(colorMsk)[:,:,:3]
  
  colorRamp, colorLabel = loadColorRampAndLabel(COLOR_INDEX)
  grayMsk, areas = colorMask2GrayMask(colorMsk, colorRamp)
  
  grayMskName = prefix+'_gray_mask.png' if prefix else 'gray_mask.png'
  sitk.WriteImage(sitk.GetImageFromArray(grayMsk), os.path.join(folder_path, grayMskName))
  
  curves = getTrendCurves(vol, grayMsk, colorRamp, average_factor)
  calciums = getTotalCalcium(vol, grayMsk, colorRamp, areas)

  table = pd.DataFrame({'Idx':colorRamp.keys(), 'Abbreviation':list(colorLabel.values()), 'Areas':list(areas.values()), 'Normalized Total':calciums})
  totalFile = prefix+'_total.csv' if prefix else 'total.csv'
  table.to_csv(os.path.join(folder_path, totalFile), index=False)
  
  # baseline percentage
  factor = 0.2

  for idx, curve, idx1 in zip(colorRamp, curves, colorLabel):
    lbl = colorLabel[idx1]
    
    if SAVE_CURVE_CSV:
      table = pd.DataFrame({'Count':curve})
      csvFile = 'final_curve_'+lbl+'.csv'
      if prefix:
        csvFile = prefix + '_' + csvFile
      table.to_csv(os.path.join(folder_path, csvFile), index=True)
    
    # find peaks
    h = np.amin(curve) + factor*(np.amax(curve)-np.amin(curve))
    peaks, _ = find_peaks(curve, height=h)
    print(idx, len(peaks), np.amin(curve), h, np.amax(curve))

    #print(peaks)
    
    plt.figure(figsize=(12,6))
    plt.plot(curve, label = str(idx) + ': ' + lbl)
    plt.plot(peaks, np.array(curve)[ peaks ], "x")
    plt.axhline(y = h, color = 'r', linestyle = '--') 
    plt.legend()
    plt.title(folder_path)
    plt.xlabel("Frames (1:" + str(args.average_factor)+")")
    plt.ylabel("Delta")
    
    chartName = 'delta_'+str(idx)+'_'+lbl+'.png'
    if prefix:
      chartName = prefix + '_' + chartName
    plt.savefig(os.path.join(folder_path, chartName))
    plt.close()
  return True
  
def main(root_folder, average_factor, prefix=""):
  """
  Main function to process the dataset.
  
  Args:
    root_folder (str): Path to the root folder containing the dataset
  """
  # Track statistics
  total_folders = 0
  processed_folders = 0
  skipped_folders = 0
  
  # Iterate through the directory structure
  for animal_folder in os.listdir(root_folder):
    animal_path = os.path.join(root_folder, animal_folder)
    
    # Skip if not a directory
    if not os.path.isdir(animal_path):
      continue
        
    print(f"Processing animal: {animal_folder}")
    
    # Process each video folder under the animal folder
    for video_folder in os.listdir(animal_path):
      video_path = os.path.join(animal_path, video_folder)
      
      # Skip if not a directory
      if not os.path.isdir(video_path):
        continue
            
      total_folders += 1
      
      # Check if the required PNG file exists
      png_file = os.path.join(video_path, COLOR_MASK)
      
      if os.path.exists(png_file):
        print(f"  Found required PNG in: {video_folder}")
        
        # Process the data in this folder and write results to the same folder
        if process_data_in_folder(video_path, average_factor, prefix):
          processed_folders += 1
      else:
        print(f"  Skipping {video_folder} - required PNG not found")
        skipped_folders += 1
  
  # Print summary
  print("\nProcessing Summary:")
  print(f"Total folders examined: {total_folders}")
  print(f"Folders processed: {processed_folders}")
  print(f"Folders skipped: {skipped_folders}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Trend graph generator based on aligned atlas and region definition")
  parser.add_argument("--root_folder", required=True, help="Root folder containing the dataset")
  parser.add_argument("--average_factor", type=int, default=600, help="frame average factor, for example, 600 will average 20 seconds if fps is 30")
  parser.add_argument("--prefix", default="", help="Optional prefix for output filenames")
  args = parser.parse_args()
  
  # Validate that average_factor is greater than 0
  if args.average_factor <= 0:
    parser.error("--average_factor must be a positive integer")
    
  print(args, flush=True)
  
  print(f"Starting to process dataset at: {args.root_folder}")
  if args.prefix:
    print(f"Using prefix for output files: '{args.prefix}'")
  main(args.root_folder, args.average_factor, args.prefix)
  print("Processing complete!")
