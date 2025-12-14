import os
import subprocess
import shutil
import time
from pathlib import Path

# Dataset and configuration settings
DATASET_NAME = "fern"
DATASET_VIEWS = "3v"  # Number of views used
BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "comparison_results"
TEMP_RENDER_DIR = BASE_DIR / "temp_renders"

# Define methods to compare, each with its own config file and CLI arguments
METHODS = {
    "sparsenerf": {
        "config": "configs/fern_3v.txt",
        "args": [
            "--use_dpt_ranking",
            "--no_batching",
            "--N_iters", "1000",
            "--lrate", "0.01",
            "--lrate_decay", "10",
            "--lambda_rank", "0.2",
            "--lambda_cont", "0.02",
            "--render_factor", "8",
            "--i_video", "1000"
        ]
    },
    "dsnerf": {
        "config": "configs/fern_3v_ds.txt",
        "args": [
            "--finest_res", "1024",
            "--log2_hashmap_size", "19",
            "--lrate", "0.01",
            "--lrate_decay", "10",
            "--render_factor", "8",
            "--i_video", "1000"
        ]
    },
    "hashnerf": {
        "config": "configs/fern_3v_hash.txt",
        "args": [
            "--render_factor", "8",
            "--i_video", "1000"
        ]
    }
}

def ensure_dir(path):
    """Ensure that the given directory path exists (create if missing)."""
    path.mkdir(parents=True, exist_ok=True)

def run_command(cmd, cwd=None):
    """
    Run a shell command and raise an error if it fails.
    
    Args:
        cmd (list or str): Command to execute.
        cwd (str, optional): Working directory for the command.
    """
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def download_fern_if_needed():
    """Download the 'fern' dataset if it doesn't already exist."""
    data_path = DATA_DIR / "nerf_llff_data" / DATASET_NAME
    if not data_path.exists():
        print("Downloading 'fern' dataset...")
        run_command(["python", "download_dataset.py", "--dataset", DATASET_NAME])

def render_method(name, config, args):
    """
    Train and render a NeRF method, then copy outputs to a dedicated results folder.
    
    Args:
        name (str): Name of the method (e.g., 'sparsenerf').
        config (str): Path to config file.
        args (list): Additional command-line arguments.
    """
    exp_dir = OUTPUT_ROOT / name
    ensure_dir(exp_dir)
    
    # Build and run the training command
    cmd = ["python", "run_nerf.py", "--config", str(config)] + [str(a) for a in args]
    run_command(cmd)

    # Locate the most recent log directory for this dataset
    logs_dir = BASE_DIR / "logs"
    possible_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and DATASET_NAME in d.name]
    if not possible_dirs:
        raise FileNotFoundError(f"No log directory found for method: {name}")
    latest_log = max(possible_dirs, key=os.path.getmtime)

    # Copy logs and outputs to the experiment directory
    shutil.copytree(latest_log, exp_dir / "logs", dirs_exist_ok=True)

    video_src = latest_log / "video.mp4"
    depths_src = latest_log / "depths"
    if video_src.exists():
        shutil.copy(video_src, exp_dir / f"{name}_video.mp4")
    if depths_src.exists():
        shutil.copytree(depths_src, exp_dir / "depth_maps", dirs_exist_ok=True)

def create_annotated_comparison_video():
    """
    Create a side-by-side comparison video of all methods.
    Each frame shows outputs from sparsenerf, dsnerf, and hashnerf with labels.
    """
    import cv2
    import numpy as np

    # Paths to individual method videos
    videos = {
        "sparsenerf": OUTPUT_ROOT / "sparsenerf" / "sparsenerf_video.mp4",
        "dsnerf": OUTPUT_ROOT / "dsnerf" / "dsnerf_video.mp4",
        "hashnerf": OUTPUT_ROOT / "hashnerf" / "hashnerf_video.mp4"
    }

    # Open video captures
    caps = {}
    for name, path in videos.items():
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        caps[name] = cv2.VideoCapture(str(path))

    # Get video properties from the first video (assumed consistent across all)
    cap_ref = list(caps.values())[0]
    width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_ref.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video: 3x width (side-by-side)
    out_width = width * 3
    out_height = height
    out_path = OUTPUT_ROOT / "comparison_all_methods.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (out_width, out_height))

    # Process each frame
    for i in range(total_frames):
        frames = []
        for name in ["sparsenerf", "dsnerf", "hashnerf"]:
            ret, frame = caps[name].read()
            if not ret:
                # If frame missing, use black placeholder
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # Annotate with method name
                cv2.putText(frame, name.upper(), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(frame)

        combined = np.hstack(frames)
        out.write(combined)

    # Clean up
    for cap in caps.values():
        cap.release()
    out.release()
    print(f"Comparison video saved: {out_path}")

if __name__ == "__main__":
    # Ensure output directories exist
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(TEMP_RENDER_DIR)

    # Download dataset if missing
    download_fern_if_needed()

    # Run each method sequentially
    for name, cfg in METHODS.items():
        print(f"\nRunning {name}...")
        render_method(name, cfg["config"], cfg["args"])
        time.sleep(2)  # Small delay between runs

    # Attempt to generate comparison video
    try:
        create_annotated_comparison_video()
    except ImportError:
        print("   OpenCV not installed. Skipping comparison video generation.")
        print("   Install it via: pip install opencv-python")

    print(f"\nAll results saved in: {OUTPUT_ROOT}")