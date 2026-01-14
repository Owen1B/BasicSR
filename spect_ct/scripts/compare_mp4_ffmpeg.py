#!/usr/bin/env python3
"""
Compare MP4 visualizations from two models side-by-side using ffmpeg (better compatibility).

Usage:
    python spect_ct/scripts/compare_mp4_ffmpeg.py \
        --model1_dir experiments/n2n_spect229_60view3d_unet_projonly_full_multidose/visualization/mp4s \
        --model2_dir experiments/n2n_spect229_60view3d_unet_projonly_full_multidose_L_pretrain_from_Lk2_91k/visualization/mp4s \
        --output_dir results/comparison_mp4s \
        --model1_name "Base Multidose" \
        --model2_name "L Multidose"
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


def find_latest_mp4(base_dir: Path, patient_name: str):
    """Find the latest MP4 file for a patient."""
    patient_dir = base_dir / patient_name
    if not patient_dir.exists():
        return None
    
    mp4_files = sorted(patient_dir.glob("*.mp4"))
    if len(mp4_files) == 0:
        return None
    
    # Return the latest one (by modification time or filename)
    return max(mp4_files, key=lambda p: p.stat().st_mtime)


def create_side_by_side_with_ffmpeg(video1_path: Path, video2_path: Path, output_path: Path, 
                                    model1_name: str, model2_name: str):
    """Create side-by-side video using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg filter_complex to create side-by-side video with labels
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video1_path),
        '-i', str(video2_path),
        '-filter_complex',
        f'[0:v]scale=iw:ih,setsar=1,drawtext=text=\'{model1_name}\':fontcolor=white:fontsize=24:x=10:y=10[v0];'
        f'[1:v]scale=iw:ih,setsar=1,drawtext=text=\'{model2_name}\':fontcolor=white:fontsize=24:x=10:y=10[v1];'
        f'[v0][v1]hstack=inputs=2[v]',
        '-map', '[v]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: ffmpeg failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"ffmpeg stderr: {e.stderr.decode()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare MP4 visualizations from two models using ffmpeg")
    parser.add_argument("--model1_dir", type=str, required=True, help="Directory containing model1 MP4s")
    parser.add_argument("--model2_dir", type=str, required=True, help="Directory containing model2 MP4s")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for comparison videos")
    parser.add_argument("--model1_name", type=str, default="Model1", help="Label for model1")
    parser.add_argument("--model2_name", type=str, default="Model2", help="Label for model2")
    
    args = parser.parse_args()
    
    model1_dir = Path(args.model1_dir)
    model2_dir = Path(args.model2_dir)
    output_dir = Path(args.output_dir)
    
    if not model1_dir.exists():
        raise ValueError(f"Model1 directory does not exist: {model1_dir}")
    if not model2_dir.exists():
        raise ValueError(f"Model2 directory does not exist: {model2_dir}")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed. Please install it first:")
        print("  sudo apt-get update && sudo apt-get install -y ffmpeg")
        return
    
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
        # Find latest MP4 files
        mp41 = find_latest_mp4(model1_dir, patient)
        mp42 = find_latest_mp4(model2_dir, patient)
        
        if mp41 is None or mp42 is None:
            print(f"Warning: Missing MP4 files for patient {patient}, skipping.")
            continue
        
        print(f"\nProcessing {patient}:")
        print(f"  Model1: {mp41.name}")
        print(f"  Model2: {mp42.name}")
        
        # Create output filename
        output_filename = f"{patient}_comparison.mp4"
        output_path = output_dir / output_filename
        
        # Create side-by-side video
        success = create_side_by_side_with_ffmpeg(
            mp41, mp42, output_path,
            model1_name=args.model1_name,
            model2_name=args.model2_name
        )
        
        if success:
            print(f"  Saved: {output_path}")
        else:
            print(f"  Failed to create video for {patient}")


if __name__ == "__main__":
    main()

