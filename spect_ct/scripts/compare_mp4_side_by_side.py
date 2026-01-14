#!/usr/bin/env python3
"""
Compare MP4 visualizations from two models side-by-side.

Usage:
    python spect_ct/scripts/compare_mp4_side_by_side.py \
        --model1_dir results/test_n2n_spect229_60view3d_unet_projonly_full_multidose_mp4/visualization/mp4s \
        --model2_dir results/test_n2n_spect229_60view3d_unet_projonly_full_multidose_L_pretrain_from_Lk2_91k_mp4/visualization/mp4s \
        --output_dir results/comparison_mp4s \
        --model1_name "Base Multidose" \
        --model2_name "L Multidose"
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def find_mp4_files(base_dir: Path, patient_name: str, thin_factor: int = None):
    """Find MP4 files for a patient, optionally filtered by thin_factor."""
    patient_dir = base_dir / patient_name
    if not patient_dir.exists():
        return []
    
    mp4_files = list(patient_dir.glob("*.mp4"))
    if thin_factor is not None:
        # Filter by thin_factor in filename (e.g., "thin2345" or "thin2")
        mp4_files = [f for f in mp4_files if f"thin{thin_factor}" in f.stem or f"thin2345" in f.stem]
    
    return sorted(mp4_files)


def extract_frames(video_path: Path):
    """Extract all frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def create_side_by_side_video(frames1, frames2, output_path: Path, fps=10, model1_name="Model1", model2_name="Model2"):
    """Create a side-by-side comparison video."""
    if len(frames1) == 0 or len(frames2) == 0:
        print(f"Warning: Empty frames for {output_path.name}, skipping.")
        return False
    
    # Ensure same number of frames (pad or truncate to shorter)
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    # Get dimensions
    h1, w1 = frames1[0].shape[:2]
    h2, w2 = frames2[0].shape[:2]
    
    # Resize to same height if needed
    target_h = max(h1, h2)
    if h1 != target_h:
        frames1 = [cv2.resize(f, (int(w1 * target_h / h1), target_h)) for f in frames1]
    if h2 != target_h:
        frames2 = [cv2.resize(f, (int(w2 * target_h / h2), target_h)) for f in frames2]
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    
    # Create side-by-side frames
    combined_frames = []
    for f1, f2 in zip(frames1, frames2):
        # Add labels
        f1_labeled = f1.copy()
        f2_labeled = f2.copy()
        cv2.putText(f1_labeled, model1_name, (10, 30), font, font_scale, color, thickness)
        cv2.putText(f2_labeled, model2_name, (10, 30), font, font_scale, color, thickness)
        
        # Concatenate horizontally
        combined = np.hstack([f1_labeled, f2_labeled])
        combined_frames.append(combined)
    
    # Write video using ffmpeg for best compatibility
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save frames as temporary images, then use ffmpeg to encode
    temp_dir = tempfile.mkdtemp(prefix='mp4_compare_')
    try:
        # Write frames as images
        frame_files = []
        for i, frame in enumerate(combined_frames):
            frame_file = Path(temp_dir) / f'frame_{i:06d}.png'
            cv2.imwrite(str(frame_file), frame)
            frame_files.append(frame_file)
        
        # Use ffmpeg to create video from images
        try:
            # Build ffmpeg command
            input_pattern = str(Path(temp_dir) / 'frame_%06d.png')
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', input_pattern,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Fallback to cv2 if ffmpeg not available
            print(f"Warning: ffmpeg not available, using cv2 fallback: {e}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (combined_frames[0].shape[1], combined_frames[0].shape[0]))
            
            if not out.isOpened():
                print(f"Error: Could not open video writer for {output_path}")
                return False
            
            for frame in combined_frames:
                out.write(frame)
            out.release()
            return True
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Compare MP4 visualizations from two models")
    parser.add_argument("--model1_dir", type=str, required=True, help="Directory containing model1 MP4s")
    parser.add_argument("--model2_dir", type=str, required=True, help="Directory containing model2 MP4s")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for comparison videos")
    parser.add_argument("--model1_name", type=str, default="Model1", help="Label for model1")
    parser.add_argument("--model2_name", type=str, default="Model2", help="Label for model2")
    parser.add_argument("--thin_factor", type=int, default=None, help="Filter by thin_factor (e.g., 2, 3, 4, 5). If None, use all.")
    parser.add_argument("--fps", type=int, default=10, help="FPS for output video")
    
    args = parser.parse_args()
    
    model1_dir = Path(args.model1_dir)
    model2_dir = Path(args.model2_dir)
    output_dir = Path(args.output_dir)
    
    if not model1_dir.exists():
        raise ValueError(f"Model1 directory does not exist: {model1_dir}")
    if not model2_dir.exists():
        raise ValueError(f"Model2 directory does not exist: {model2_dir}")
    
    # Find all patient directories
    patients1 = {d.name for d in model1_dir.iterdir() if d.is_dir()}
    patients2 = {d.name for d in model2_dir.iterdir() if d.is_dir()}
    common_patients = sorted(patients1 & patients2)
    
    if len(common_patients) == 0:
        print(f"Warning: No common patients found between {model1_dir} and {model2_dir}")
        return
    
    print(f"Found {len(common_patients)} common patients: {common_patients}")
    
    # Process each patient
    for patient in tqdm(common_patients, desc="Processing patients"):
        # Find MP4 files
        mp4s1 = find_mp4_files(model1_dir, patient, args.thin_factor)
        mp4s2 = find_mp4_files(model2_dir, patient, args.thin_factor)
        
        if len(mp4s1) == 0 or len(mp4s2) == 0:
            print(f"Warning: No MP4 files found for patient {patient}, skipping.")
            continue
        
        # Match MP4s by thin_factor (if specified) or use first available
        # For now, use the first MP4 from each model
        mp41 = mp4s1[0]
        mp42 = mp4s2[0]
        
        # If thin_factor specified, try to match more precisely
        if args.thin_factor is not None:
            # Prefer exact match
            mp41_matched = [f for f in mp4s1 if f"thin{args.thin_factor}" in f.stem]
            mp42_matched = [f for f in mp4s2 if f"thin{args.thin_factor}" in f.stem]
            if len(mp41_matched) > 0 and len(mp42_matched) > 0:
                mp41 = mp41_matched[0]
                mp42 = mp42_matched[0]
        
        print(f"\nProcessing {patient}:")
        print(f"  Model1: {mp41.name}")
        print(f"  Model2: {mp42.name}")
        
        # Extract frames
        frames1 = extract_frames(mp41)
        frames2 = extract_frames(mp42)
        
        if len(frames1) == 0 or len(frames2) == 0:
            print(f"  Warning: Empty video, skipping.")
            continue
        
        # Create output filename
        thin_suffix = f"_thin{args.thin_factor}" if args.thin_factor is not None else ""
        output_filename = f"{patient}_comparison{thin_suffix}.mp4"
        output_path = output_dir / output_filename
        
        # Create side-by-side video
        success = create_side_by_side_video(
            frames1, frames2, output_path, 
            fps=args.fps,
            model1_name=args.model1_name,
            model2_name=args.model2_name
        )
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed to create video for {patient}")


if __name__ == "__main__":
    main()

