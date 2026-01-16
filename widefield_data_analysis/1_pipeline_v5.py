"""
2025-05-19
Do Global SVD, detrend, and band pass filtering (Steinmetz et al. style)
https://www.nature.com/articles/s41586-019-1787-x
"""

import os
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np

from scipy.signal import detrend
from scipy.signal import butter, sosfilt
from sklearn.utils.extmath import randomized_svd

import SimpleITK as sitk

import time
import sys

def loadMovie(movieName):
  videoStream = cv2.VideoCapture(movieName)
  fps = videoStream.get(cv2.CAP_PROP_FPS)
  print('Frames', videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
  print('FPS', fps)
  
  limit = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))

  stack = []
  for ii in range(limit):
    ret, frame = videoStream.read()
    # Convert the BGR frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # down size by half to make it easier to process
    resized_grayscale = cv2.resize(grayscale_frame,
                                   (grayscale_frame.shape[1] // 2, grayscale_frame.shape[0] // 2),
                                   interpolation=cv2.INTER_AREA) # Added interpolation for better resize quality
    stack.append(resized_grayscale)

  videoStream.release()
  
  return np.array(stack).astype(np.float32), fps

def computeBaseFrame(stack):
  """
  Calculates F0 as the average of the lowest 20% of values for each pixel's time series.
  Vectorized approach using sorting.

  Args:
    stack (np.ndarray): Input 3D stack (time, height, width).

  Returns:
    np.ndarray: 2D F0 array (height, width).
  """
  n_time, height, width = stack.shape

  # Reshape stack to (time, n_pixels) to operate on each pixel's time series
  stack_reshaped = stack.reshape(n_time, height * width)

  # Sort values along the time axis (axis=0) for each pixel
  sorted_stack = np.sort(stack_reshaped, axis=0)

  # Determine the number of time points that constitute the lowest 20%
  k = max(1, int(np.ceil(0.20 * n_time))) # make sure it is at least 1

  # Select the lowest k values for each pixel
  lowest_20_percent_values = sorted_stack[:k, :] # Shape: (k, n_pixels)

  # Calculate the mean of these lowest values for each pixel
  f0_values_flat = np.mean(lowest_20_percent_values, axis=0) # Shape: (n_pixels,)

  # Reshape F0 back to (height, width)
  f0_map = f0_values_flat.reshape(height, width)
    
  return f0_map

def DFoF(stack):
  """
  use the average of buttom 20% values as the base to compute delta_f over f.
  this is done pixel wise
  """
  baseFrame = computeBaseFrame(stack)
  print("base frame (avg of lowest 20%)", baseFrame.shape, np.amin(baseFrame), np.amax(baseFrame), flush=True)
  
  e = 1e-9
  newStack = []
  for frame in stack:
    newFrame = frame - baseFrame
    
    # Use np.divide with where parameter to safely divide
    result = np.divide(newFrame, 
                       baseFrame, 
                       out=np.zeros_like(newFrame, dtype=np.float32), 
                       where=baseFrame>e)

    newStack.append(result)

  return np.array(newStack)

def SVD_global(stack_3d, rank, n_iter=5, random_seed=2025):
  """
  Performs SVD on the entire 3D stack (reshaped) as per Steinmetz et al. (2019).
  The temporal components (columns of V) are returned directly for further processing.

  Args:
    stack_3d (np.ndarray): The input 3D stack (time, height, width).
    rank (int): The number of singular values/components to keep.
    n_iter (int): Number of power iterations for randomized SVD
    random_seed (int): random seed
  Returns:
    U_trunc (np.ndarray): Truncated spatial components (pixels, rank).
    s_trunc (np.ndarray): Truncated singular values (rank,).
    Vh_trunc_processed (np.ndarray): Original (unprocessed) truncated temporal components (rank, time).
                                     This Vh is V.T from M = USVt.
    original_shape (tuple): Original shape (time, height, width) for reshaping U.
    num_pixels (int): Total number of pixels (height * width).
  """
  original_shape = stack_3d.shape
  n_frames, height, width = original_shape
  num_pixels = height * width

  # Reshape to (pixels, time)
  # stack_3d is (time, height, width), so stack_3d.reshape(n_frames, num_pixels) is (time, pixels)
  M_pixels_time = stack_3d.reshape(n_frames, num_pixels).T
  print(f"Reshaped M for SVD: {M_pixels_time.shape} (pixels, time)")

  print(f"Performing randomized SVD for rank {rank}...")
  # Using sklearn's randomized_svd
  # U: (pixels, rank)
  # s: (rank,)
  # Vh: (rank, time)
  # n_components is the 'rank' we want
  # n_oversamples is usually set to something like 10 or 2*rank for stability
  # n_iter controls the number of power iterations for accuracy
  U_trunc, s_trunc, Vh_trunc = randomized_svd(
      M_pixels_time,
      n_components=rank,
      n_iter=n_iter, # e.g., 5 or 7. Default is 'auto' which is often good.
      n_oversamples=max(10, 2 * rank), # Add some oversampling
      random_state=random_seed # For reproducibility
  )

  print(f"Randomized SVD components shapes: U={U_trunc.shape}, s={s_trunc.shape}, Vh={Vh_trunc.shape}")

  return U_trunc, s_trunc, Vh_trunc, original_shape, num_pixels

def reconstruct_from_SVD_components(U_trunc, s_trunc, Vh_processed_trunc, original_shape, num_pixels):
  """
  Reconstructs the 3D stack from processed SVD components.

  Args:
    U_trunc (np.ndarray): Truncated spatial components (pixels, rank).
    s_trunc (np.ndarray): Truncated singular values (rank,).
    Vh_processed_trunc (np.ndarray): Processed (e.g., detrended, filtered)
                                     truncated temporal components (rank, time).
    original_shape (tuple): Original shape (time, height, width) of the stack.
    num_pixels (int): Total number of pixels (height * width).

  Returns:
    reconstructed_stack_3d (np.ndarray): The reconstructed 3D stack (time, height, width).
  """
  n_frames, height, width = original_shape

  # Reconstruct M_reconstructed = U_trunc @ np.diag(s_trunc) @ Vh_processed_trunc
  # This results in a (pixels, time) matrix
  M_reconstructed_pixels_time = U_trunc @ np.diag(s_trunc) @ Vh_processed_trunc
  print(f"Reconstructed M: {M_reconstructed_pixels_time.shape} (pixels, time)")

  # Reshape back to (time, pixels) and then to (time, height, width)
  # M_reconstructed_pixels_time.T gives (time, pixels)
  reconstructed_stack_3d = M_reconstructed_pixels_time.T.reshape(n_frames, height, width)
  print(f"Reshaped reconstructed stack: {reconstructed_stack_3d.shape}")

  return reconstructed_stack_3d
  
def SVD(stack, rank):
  newStack = []
  for frame in stack:
    U, s, Vh = np.linalg.svd(frame, full_matrices=False)
    s_filtered = np.copy(s)
    s_filtered[rank:] = 0
    # reconstructed frame
    frame_r = U @ np.diag(s_filtered) @ Vh
    newStack.append(frame_r)

  newStack = np.array(newStack)
  return newStack

def butter_bandpass(lowcut, highcut, fs, order=2):
  nyq = 0.5 * fs
  print('nyq:', nyq, 'fps:', fs)
  low = lowcut / nyq
  high = highcut / nyq
  sos = butter(order, [low, high], analog=False, btype='bandstop', output='sos', fs=fs)
  return sos

def butter_highpass(fq, fs, order=2):
  nyq = 0.5 * fs
  print('nyq:', nyq, 'fps:', fs)
  sos = butter(order, fq, analog=False, btype='highpass', output='sos', fs=fs)
  return sos

def butter_bandpass_filter(data, axis, lowcut, highcut, fs, order=2):
  sos = butter_bandpass(lowcut, highcut, fs, order=order)
  y = sosfilt(sos, data, axis=axis)
  return y

def butter_highpass_filter(data, axis, fq, fs, order=2):
  sos = butter_highpass(fq, fs, order=order)
  y = sosfilt(sos, data, axis=axis)
  return y
  
def process_video(video_path, output_dir):
  INPUT = video_path
  OUTPUT = output_dir
  
  # load channel images
  print('*** data loading ***')
  tStart = time.time()
  signalChannelNpy, fps = loadMovie(INPUT)
  print(os.path.basename(video_path))
  print(f"Loaded in {(time.time()-tStart):.2f} seconds.", 
        'Signal:', signalChannelNpy.shape, 
        'FPS:', fps, 
        np.amin(signalChannelNpy), np.amax(signalChannelNpy),
        flush=True)

  # compute DFoF
  print('*** DFoF ***')
  tStart = time.time()
  dfof = DFoF(signalChannelNpy)
  if args.save_steps:
    sitk.WriteImage(sitk.GetImageFromArray(dfof.astype(np.float32)), 
                    os.path.join(OUTPUT, "dfof_ch1.nrrd"))
  print(f"Done in {(time.time()-tStart):.2f} seconds.", np.amin(dfof), np.amax(dfof), flush=True)
  
  print('*** Global SVD, Detrending, Filtering (Steinmetz et al. style) ***')
  SVD_RANK = 50 

  # Perform global SVD
  # U_trunc: (pixels, rank), s_trunc: (rank,), Vh_trunc: (rank, time)
  U_spatial_trunc, s_values_trunc, Vh_temporal_trunc, original_img_shape, n_pixels = SVD_global(dfof, SVD_RANK)

  V_temporal_components = Vh_temporal_trunc.T # Shape: (time, rank)
  print(f"Temporal components V (for processing): {V_temporal_components.shape} (time, rank)", flush=True)

  # Detrend temporal components (columns of V)
  print('*** Detrending temporal components (V) ***')
  V_detrended = np.zeros_like(V_temporal_components)
  V_detrended = detrend(V_temporal_components, axis=0) # Linear detrend by default

  # Filter temporal components (columns of V_detrended)
  print('*** Heart beat filtering temporal components (V_detrended) ***', flush=True)
  V_heartbeat_filtered = np.zeros_like(V_detrended)
  V_heartbeat_filtered = butter_bandpass_filter(V_detrended,
                                                axis=0, # Filter along time axis
                                                lowcut=7, highcut=14,
                                                fs=fps,
                                                order=2)


  print('*** Highpass filtering temporal components (V_heartbeat_filtered) ***')
  V_fully_filtered = np.zeros_like(V_heartbeat_filtered)
  V_fully_filtered = butter_highpass_filter(V_heartbeat_filtered,
                                            axis=0, # Filter along time axis
                                            fq=0.01,
                                            fs=fps,
                                            order=2)

  # V_fully_filtered has shape (time, rank).
  # transpose it back to (rank, time) to match Vh_trunc format for reconstruction.
  Vh_processed_trunc = V_fully_filtered.T # Shape: (rank, time)

  # Reconstruct data using processed temporal components and original spatial components
  print('*** Reconstructing DFOF stack from processed SVD components ***', flush=True)
  dfof_reconstructed = reconstruct_from_SVD_components(U_spatial_trunc, s_values_trunc, Vh_processed_trunc,
                                                      original_img_shape, n_pixels)
  
  print(f"Final reconstructed stack shape: {dfof_reconstructed.shape}")
  print(f"Done in {(time.time()-tStart):.2f} seconds.", np.amin(dfof_reconstructed), np.amax(dfof_reconstructed), flush=True)
  
  # TODO: check to see if clipping is necessary
  #dfof_reconstructed[dfof_reconstructed < 0] = 0

  sitk.WriteImage(sitk.GetImageFromArray(dfof_reconstructed.astype(np.float32)),
                    os.path.join(OUTPUT, "final.nrrd"))

  print(f"Processed: {video_path}", flush=True)

def main(data_root_folder, output_root_folder):
  # Ensure the output root folder exists
  os.makedirs(output_root_folder, exist_ok=True)
  
  # Get all MP4 files in the directory structure
  mp4_pattern = os.path.join(data_root_folder, "*", "*", "*.mp4")
  mp4_files = sorted(glob.glob(mp4_pattern))
  
  # skip non 'SS' videos
  mp4_files = [v for v in mp4_files if 'SS' in v]
  print(f"Found {len(mp4_files)} MP4 files to process", flush=True)
  
  for mp4_file in mp4_files:
    # Get relative path components
    rel_path = os.path.relpath(mp4_file, data_root_folder)
    animal_folder = os.path.dirname(os.path.dirname(rel_path))
    day_folder = os.path.dirname(os.path.basename(mp4_file))
    file_name = os.path.splitext(os.path.basename(mp4_file))[0]
    
    # Create the output directory structure
    output_dir = os.path.join(output_root_folder, animal_folder, day_folder, file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the video and save results to the output directory
    process_video(mp4_file, output_dir)

if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description="Main processing pipeline")
  parser.add_argument("--data_root", required=True, help="Root folder containing the data")
  parser.add_argument("--output_root", required=True, help="Root folder for output results")
  parser.add_argument('--save_steps', action="store_true", help="save intermediate results")
  args = parser.parse_args()
  
  print(args, flush=True)
  
  main(args.data_root, args.output_root)
  
  print('done.')
