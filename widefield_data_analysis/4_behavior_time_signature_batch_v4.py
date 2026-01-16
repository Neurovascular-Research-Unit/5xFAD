import cv2
import numpy as np
import argparse
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def estimate_threshold_with_otsu(pixel_counts):
    """
    Estimates a motion area threshold using Otsu's binarization method.
    """
    try:
        upper_bound = int(np.percentile(pixel_counts, 99.9))
        if upper_bound == 0:
             print("INFO: No significant motion pixel counts found to analyze.")
             return 100
        
        clipped_counts = np.clip(pixel_counts, 0, upper_bound)
        normalized_counts = (255 * (clipped_counts / upper_bound)).astype(np.uint8)
        otsu_thresh_normalized, _ = cv2.threshold(normalized_counts, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh_original_scale = int(otsu_thresh_normalized * (upper_bound / 255.0))
        
        print(f"--- Otsu's algorithm detected threshold at: {otsu_thresh_original_scale} pixels.")
        return otsu_thresh_original_scale
    except Exception as e:
        print(f"--- Otsu's algorithm failed with an error: {e}")
        return None

def merge_close_events(events, max_gap_frames):
    """
    Merges sequential motion events that have a small gap between them.

    Args:
        events (list): A list of event dictionaries.
        max_gap_frames (int): The maximum number of frames between events to be merged.

    Returns:
        list: A new list of merged event dictionaries.
    """
    if len(events) < 2:
        return events

    merged_events = []
    current_event = events[0]

    for next_event in events[1:]:
        gap = next_event['start_frame'] - current_event['end_frame']
        if gap <= max_gap_frames:
            # Merge by extending the end_frame of the current event
            current_event['end_frame'] = next_event['end_frame']
        else:
            # Gap is too large, finalize the current event and start a new one
            merged_events.append(current_event)
            current_event = next_event
    
    # Add the last event
    merged_events.append(current_event)
    
    return merged_events

def plot_motion_histogram(pixel_counts, threshold, video_filename, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(pixel_counts, bins=200, log=True, color='gray')
    ax.axvline(x=threshold, color='lime', linestyle='--', linewidth=2, label=f'Threshold Used = {threshold}')
    ax.set_title(f"Motion Pixel Count Histogram for: {video_filename}")
    ax.set_xlabel("Number of Motion Pixels in Frame Difference")
    ax.set_ylabel("Frame Count (Log Scale)")
    ax.legend()
    ax.grid(axis='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"-> Histogram plot saved to: {output_path}")

def plot_motion_timeline(total_frames, motion_events, video_filename, output_path):
    fig, ax = plt.subplots(figsize=(15, 2.5))
    motion_color, stationary_color = 'tab:red', 'tab:blue'
    y_val, height = (0, 1)
    ax.broken_barh([(0, total_frames)], (y_val, height), facecolors=stationary_color)
    if motion_events:
        motion_ranges = [(e['start_frame'] - 1, e['end_frame'] - e['start_frame'] + 1) for e in motion_events]
        ax.broken_barh(motion_ranges, (y_val, height), facecolors=motion_color)
    ax.set_xlim(0, total_frames)
    ax.set_ylim(y_val, y_val + height)
    ax.set_xlabel("Frame Number")
    ax.set_yticks([])
    ax.set_title(f"Motion Timeline for: {video_filename}", pad=15)
    motion_patch = mpatches.Patch(color=motion_color, label='Motion Event')
    stationary_patch = mpatches.Patch(color=stationary_color, label='Stationary')
    ax.legend(handles=[motion_patch, stationary_patch], bbox_to_anchor=(0.5, 1.4), loc='lower center', ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"-> Timeline plot saved to: {output_path}")

def save_motion_event_frames(video_path, diff_thresh, events, output_dir):
    frames_to_save = {i for event in events for i in range(event['start_frame'], event['end_frame'] + 1)}
    if not frames_to_save: 
        return
    print(f"--- Saving {len(frames_to_save)} motion frames with solid red overlays...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print("ERROR: Could not re-open video to save frames."); 
        return

    prev_frame_gray_blur, frame_idx, saved_count = None, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_idx += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if frame_idx in frames_to_save:
            if prev_frame_gray_blur is None: 
                prev_frame_gray_blur = gray_blur; 
                continue
            
            frame_delta = cv2.absdiff(prev_frame_gray_blur, gray_blur)
            thresh_img = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]
            thresh_img = cv2.dilate(thresh_img, None, iterations=2)
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            visualized_frame = frame.copy()
            pixel_count = cv2.countNonZero(thresh_img)
            
            cv2.drawContours(visualized_frame, contours, -1, (0, 0, 255), -1)
            
            height, width, _ = visualized_frame.shape
            downsampled_frame = cv2.resize(visualized_frame, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
            
            frame_num_str = f"{frame_idx:06d}"
            filename = f"{frame_num_str}-{pixel_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), downsampled_frame)
            saved_count += 1
            
        prev_frame_gray_blur = gray_blur

    cap.release()
    print(f"-> Successfully saved {saved_count} visualized frames.")

def analyze_and_get_pixel_counts(video_path, diff_thresh, max_time_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return None, None, None
    fps, total_frames_in_video = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_frame_limit = total_frames_in_video
    if fps > 0 and max_time_sec > 0:
        max_proc_frame = int(max_time_sec * fps)
        if total_frames_in_video > max_proc_frame: 
            effective_frame_limit = max_proc_frame
    print(f"--- Performing pass to collect pixel data from {effective_frame_limit} frames...")
    all_pixel_counts, prev_frame_gray_blur, frame_idx = [], None, 0
    while frame_idx < effective_frame_limit:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_frame_gray_blur is None:
            prev_frame_gray_blur = gray_blur
            all_pixel_counts.append(0)
            continue
        frame_delta = cv2.absdiff(prev_frame_gray_blur, gray_blur)
        thresh = cv2.threshold(frame_delta, diff_thresh, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        all_pixel_counts.append(cv2.countNonZero(thresh))
        prev_frame_gray_blur = gray_blur
    cap.release()
    return all_pixel_counts, fps, effective_frame_limit

def find_events_from_counts(pixel_counts, min_motion_area, min_event_frames, fps):
    motion_events_list, total_motion_frames = [], 0
    consecutive_motion_frames, event_start_frame, is_in_motion_event = 0, None, False
    
    for i, pixel_count in enumerate(pixel_counts):
        current_frame_has_motion = pixel_count > min_motion_area
        if current_frame_has_motion: 
            total_motion_frames += 1
        if current_frame_has_motion:
            consecutive_motion_frames += 1
            if not is_in_motion_event: 
                is_in_motion_event, event_start_frame = True, i + 1
        else:
            if is_in_motion_event:
                if consecutive_motion_frames >= min_event_frames: 
                    motion_events_list.append({'start_frame': event_start_frame, 'end_frame': i})
                    print("Number of frames:", consecutive_motive_frames, event_start_frame/fps)
                else:
                    print("Number of frames:", consecutive_motion_frames, event_start_frame/fps, "skiped")
                is_in_motion_event, consecutive_motion_frames, event_start_frame = False, 0, None
                
    if is_in_motion_event and consecutive_motion_frames >= min_event_frames:
        motion_events_list.append({'start_frame': event_start_frame, 'end_frame': len(pixel_counts)})
    return motion_events_list, total_motion_frames

def process_dataset(root_dir, diff_thresh, min_event_frames, merge_gap, output_csv, plot_dir, max_time, save_frames):
    all_event_results = []
    if save_frames and not plot_dir: 
        print("ERROR: --save_frames requires --plot_dir."); 
        return
    if not os.path.isdir(root_dir): 
        print(f"Error: Root directory not found at {root_dir}"); 
        return
    if plot_dir: 
        os.makedirs(plot_dir, exist_ok=True); 
        print(f"Plots will be saved to: {plot_dir}")
    
    print(f"Starting batch processing in root directory: {root_dir}")
    
    for animal_id in sorted(os.listdir(root_dir)):
        animal_path = os.path.join(root_dir, animal_id)
        if not os.path.isdir(animal_path): 
            continue
        for day_id in sorted(os.listdir(animal_path)):
            day_path = os.path.join(animal_path, day_id)
            if not os.path.isdir(day_path): 
                continue
            for filename in sorted(os.listdir(day_path)):
                if filename.lower().endswith('.mp4') and os.path.isfile(os.path.join(day_path, filename)):
                    video_name_base = os.path.splitext(filename)[0]
                    behavior_video_filename = f"{video_name_base} behavior.mp4"
                    behavior_video_path = os.path.join(day_path, 'Behavior', behavior_video_filename)
                    if os.path.isfile(behavior_video_path):
                        print(f"\n{'='*80}")
                        print(f"Processing: {behavior_video_path}")
                        
                        pixel_counts, fps, processed_frames = analyze_and_get_pixel_counts(behavior_video_path, diff_thresh, max_time)
                        if pixel_counts is None: 
                            print(f"Skipping due to read error."); 
                            continue
                        
                        threshold_to_use = estimate_threshold_with_otsu(pixel_counts)
                        if threshold_to_use is None:
                            threshold_to_use = 50000 # Robust fallback value for a 250x200 region
                            print(f"WARNING: Otsu estimation failed. Using fallback threshold: {threshold_to_use}")

                        initial_events, _ = find_events_from_counts(pixel_counts, threshold_to_use, min_event_frames, fps)
                        
                        # NEW: Merge close events
                        final_events = merge_close_events(initial_events, merge_gap)
                        
                        print(f"--- Analysis Complete: Found {len(initial_events)} initial events, merged into {len(final_events)} final events.")
                        
                        if plot_dir and processed_frames > 0:
                            plot_motion_timeline(processed_frames, final_events, behavior_video_filename, os.path.join(plot_dir, f"{animal_id}_{day_id}_{video_name_base}_timeline.png"))
                            plot_motion_histogram(pixel_counts, threshold_to_use, behavior_video_filename, os.path.join(plot_dir, f"{animal_id}_{day_id}_{video_name_base}_histogram.png"))
                            
                            if save_frames:
                                frame_subfolder = os.path.join(plot_dir, f"{animal_id}_{day_id}_{video_name_base}_frames")
                                os.makedirs(frame_subfolder, exist_ok=True)
                                save_motion_event_frames(behavior_video_path, diff_thresh, final_events, frame_subfolder)

                        if not final_events: 
                            continue
                        for i, event in enumerate(final_events):
                            start_frame, end_frame = event['start_frame'], event['end_frame']
                            start_time, end_time = ((start_frame / fps), (end_frame / fps)) if fps > 0 else (0,0)
                            all_event_results.append({
                                'animal_id': animal_id, 'day_id': day_id, 'video_name': behavior_video_filename,
                                'event_number': i + 1, 'start_frame': start_frame, 'end_frame': end_frame,
                                'start_time_sec': f"{start_time}", 'end_time_sec': f"{end_time}", # don't cap decimal points here for better accuracy to seek in the raw video
                                'duration_sec': f"{end_time - start_time:.3f}",
                            })
                            
    if not all_event_results: 
        print("\nBatch processing complete. No motion events detected."); 
        return
        
    print(f"\n{'='*80}\nBatch processing complete. Found {len(all_event_results)} motion events.")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_event_results[0].keys())
            writer.writeheader()
            writer.writerows(all_event_results)
        print(f"Event timestamp report successfully saved to {output_csv}")
    except IOError as e: 
        print(f"Error writing to CSV file {output_csv}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully automatic batch video motion detection using Otsu's method.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("root_dir", help="Path to the root dataset folder.")
    parser.add_argument("--output_csv", default="motion_event_timestamps.csv", help="Path for the output CSV report.")
    parser.add_argument("--plot_dir", type=str, default=None, help="Optional. Directory to save timeline and histogram plots.")
    parser.add_argument("--save_frames", action='store_true', help="Save downsampled/overlaid motion frames. Requires --plot_dir.")
    parser.add_argument("--max_time", type=float, default=53998/30.0, help="Maximum video time in seconds to process. Default: ~1800s (30 min).")
    parser.add_argument("--diff_thresh", type=int, default=25, help="Pixel difference threshold for generating motion data. Default: 25")
    parser.add_argument("--min_event_frames", type=int, default=30, help="Minimum consecutive motion frames for an event. Default: 30")
    parser.add_argument("--merge_gap", type=int, default=10, help="Maximum number of frames between two events to merge them. Default: 10")
    args = parser.parse_args()
    process_dataset(args.root_dir, args.diff_thresh, args.min_event_frames, args.merge_gap, args.output_csv, args.plot_dir, args.max_time, args.save_frames)
