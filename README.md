# Depth-NeRF: Improving NeRF Training Quality with Depth Information

> **Course Project for "Machine Learning in Robotics"**  
> **Research Topic**: *Investigating and Comparing Strategies for Incorporating Depth Data into Neural Radiance Fields under Sparse View Conditions*

## Participants
- Artyom Oganesyan
- Polina Popova
- Semynin Alexander

## Research Overview

This project investigates how **depth information** can improve NeRF training when only **few input views** are available (e.g., 3 views of a scene). Standard NeRF suffers from geometry ambiguity and slow convergence in such settings.

We compare **two distinct depth integration strategies**, all implemented within a unified NeRF codebase to ensure fair comparison:

### Methods Used

#### 1. **SparseNeRF (Depth as a Ranking Regularization)**
- **Idea**: Uses monocular depth predictions (from DPT) not as ground truth, but as a **relative depth ranking** signal.
- **Implementation**: Adds two losses:
  - `λ_rank`: enforces correct ordering of sample points along a ray.
  - `λ_cont`: encourages smoothness in depth transitions.
- **Advantage**: Robust to absolute depth errors; works even with noisy depth maps.

#### 2. **DS-NeRF (Depth as Hard Supervision)**
- **Idea**: Treats depth maps as **ground-truth supervision** during training.
- **Implementation**: Adds an L1 loss between predicted depth (via expected termination point) and input depth map.
- **Assumption**: Input depth is reasonably accurate (e.g., from COLMAP).

#### 3. **HashNeRF (Efficient Encoding + Implicit Depth)**
- **Idea**: Uses **instant-ngp-style hash encoding** for fast feature lookup, which implicitly captures geometric structure.
- **Implementation**: No explicit depth loss; geometry emerges from high-frequency encoding + sparse views.
- **Note**: Serves as a strong **depth-free baseline** to assess the true value of depth signals.

All methods share the same ray sampling, network architecture (except hash encoding), and dataset preprocessing — ensuring a controlled comparison.

---

## Results

Below are the qualitative and quantitative results obtained on the **LLFF `fern` scene with only 3 input views**.

> These are preliminary results, final testing is currently underway, after which an update will be provided.

### Demonstration Video

![preliminary results](results/first.gif)

### Metrics

| Method        | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time (1k iters) |
|---------------|--------|--------|---------|--------------------------|
| SparseNeRF    | 22.1   | 0.85   | 0.12    | 18 min                   |
| DS-NeRF       | 23.4   | 0.88   | 0.10    | 20 min                   |
| HashNeRF      | 20.7   | 0.82   | 0.15    | 12 min                   |

---
## Project Structure
```
Depth-NeRF/
│
├── README.md                              ← Full project documentation
├── requirements.txt                       ← Python dependencies 
├── .gitignore                             ← Excludes logs, cache, weights, IDE files
├── train.sh                               ← Training script
├── run_nerf.py                            ← Main training/inference script
├── run_nerf_helpers.py                    ← Helper functions for NeRF
│
├── configs/                               ← Configuration files for all methods
│   ├── fern_3v.txt                        
│   ├── fern_3v_ds.txt                     
│   └── ....                  
│
├── scripts/                               ← Utility & data preparation scripts
│   ├── download_dataset.py                ← Downloads datasets (lego, fern, etc.)
│   ├── imgs2poses.py                      ← Runs COLMAP to estimate camera poses
│   ├── download_weights.py                ← Downloads DPT weights into ./weights/
│   ├── compare_all_methods.py             ← Compares results from all methods
│   ├── download_lego_dataset.sh           ← Helper script for LEGO dataset
│   ├── eval.py                            ← Evaluation script
│   ├── eval_metrics_script.py             ← Computes PSNR, SSIM, LPIPS metrics
│   └── install_deps.py                    ← Installs system dependencies
│
├── data/                                  ← EMPTY by default (datasets downloaded here)
│   └── (populated after: python scripts/download_dataset.py --dataset fern)
│
├── results/                               ← EMPTY by default; save your outputs here
│   ├── videos/                            ← Rendered comparison videos
│   ├── depth_maps/                        ← Predicted depth visualizations
│   └── metrics/                           ← PSNR, SSIM, LPIPS, etc.
│
├── colmap_utils/                          ← COLMAP
│   ├── __init__.py
│   ├── read_write_model.py
│   └── pose_utils.py
│
├── DPT/                                   ← [Kept as-is] DPT monocular depth model
│   ├── dpt.py
│   ├── transforms.py
│   └── ... (other DPT source files)
│
├── llff/                                  ← LLFF data loader & processing
│   ├── poses.py                           ← LLFF pose utilities
│   └── ...                                ← Other LLFF utilities 
│
├── load_dataset/                          ← Dataset loading logic
│   ├── __init__.py
│   ├── load_llff.py                       ← LLFF dataset loader
│   └── data.py
│
├── losses/                                ← Custom loss functions
│   ├── sparse_depth_loss.py               ← Sparse depth ranking loss
│   └── loss.py                            ← Base loss implementations
│
├── utils/                                 ← Helper functions
│   ├── ray_utils.py                       ← Ray sampling utilities
│   ├── dpt_utils.py                       ← DPT integration utilities
│   ├── camera_pose_visualizer.py          ← Camera pose visualization
│   ├── hash_encoding.py                   ← Hash grid encoding
│   ├── optimizer.py                       ← Custom optimizers
│   └── radam.py                           ← RAdam optimizer implementation
│
└── weights/                               ← Pretrained model weights 
    └── dpt_hybrid-midas-501f0c75.pt       ← Downloaded by scripts/download_weights.py
```

## Installation and Setup

### System Requirements
- OS: Windows / Linux / macOS
- GPU: NVIDIA GPU (recommended for training; CPU works for inference)
- Python ≥ 3.8

### Step-by-Step Installation

1. **Clone the repository**
```bash
   git clone https://github.com/plnppvsln/Depth-NeRF.git
   cd Depth-NeRF
```
2. **Create a virtual environment (recommended)**
```bash
  python -m venv venv
  source venv/bin/activate    # Linux/macOS
  venv\Scripts\activate       # Windows
```
3. **Install Python dependencies**
```bash
  pip install -r requirements.txt
```

### Running the Project

1. **Download a Dataset**
The project includes a convenient script to download pre-processed NeRF datasets.

#### List Available Datasets

To see all available datasets:

```bash
python scripts/download_dataset.py --list
```

This will display all available datasets that can be downloaded.

#### Download a Dataset

To download a specific dataset:

```bash
python scripts/download_dataset.py --dataset <dataset_name>
```

For example:

```bash
python scripts/download_dataset.py --dataset lego
python scripts/download_dataset.py --dataset fern
```

#### Available Datasets

- `shaving_set`
- `lego`
- `fern`
- `fox`

#### Custom Save Directory

By default, datasets are saved to the `./data` directory. You can specify a custom directory:

```bash
python scripts/download_dataset.py --dataset lego --save_dir /path/to/custom/directory
```

**Note:** The default save directory (`./data`) is recommended as it matches the expected project structure.

#### Help

For more information, run:
```bash
python scripts/download_dataset.py --help
```
2. **Download a Weight**
Download weight:
```bash
python scripts/download_weights.py
```

3. **Running COLMAP**

COLMAP is used to estimate camera poses from a set of images. This is necessary for custom datasets or when poses are not pre-computed.

#### Prerequisites

COLMAP will be automatically downloaded on Windows if not found. On Linux/Mac, you can either:
- Install COLMAP system-wide: `sudo apt-get install colmap` (Linux) or `brew install colmap` (Mac)
- Set the `COLMAP_BINARY` environment variable to point to your COLMAP installation
- Place a COLMAP binary in `external/colmap/`

#### Running COLMAP on a Scene

To process images and generate camera poses, use the `imgs2poses.py` script:

```bash
python scripts/imgs2poses.py <scene_directory>
```

For example:

```bash
python scripts/imgs2poses.py data/nerf_llff_data/fern
python scripts/imgs2poses.py data/nerf_custom/fox
```

### Directory Structure

The scene directory should have the following structure:

```
scene_directory/
  └── images/
      ├── image1.jpg
      ├── image2.jpg
      └── ...
```

#### Match Types

You can specify the matching algorithm using the `--match_type` parameter:

- `exhaustive_matcher` (default): Best for unordered image collections
- `sequential_matcher`: Best for video sequences or ordered images

```bash
python scripts/imgs2poses.py --match_type sequential_matcher <scene_directory>
```

#### Removing Unregistered Images

By default, all images are kept in the `images/` directory, even if COLMAP couldn't register them. However, if you want to automatically remove unregistered images to avoid mismatches between images and poses, you can use the `--remove-unregistered` flag:

```bash
python scripts/imgs2poses.py --remove-unregistered <scene_directory>
```

This will delete any images from the `images/` directory that are not listed in `view_imgs.txt` (i.e., images that COLMAP couldn't register). This is useful to prevent errors when loading data, as the number of images will match the number of poses in `poses_bounds.npy`.

#### Output

After running COLMAP, the following files will be created in your scene directory:

- `database.db`: COLMAP feature database
- `sparse/0/`: Sparse reconstruction containing camera poses
  - `cameras.bin`: Camera parameters
  - `images.bin`: Image poses and metadata
  - `points3D.bin`: 3D point cloud
- `poses_bounds.npy`: Processed poses in NeRF format
- `colmap_output.txt`: COLMAP processing logs
- `view_imgs.txt`: List of successfully registered images

The script will automatically skip COLMAP if the sparse reconstruction already exists.

**Note:** COLMAP may not be able to register all images in your dataset (e.g., due to insufficient features, poor image quality, or lack of overlap). The pose processing pipeline has been updated to handle this gracefully by using only the images that were successfully registered by COLMAP. The `view_imgs.txt` file contains the list of images that were successfully processed, and the `poses_bounds.npy` file will only contain poses for these registered images. This prevents errors when some images are missing from the COLMAP reconstruction.

3. **Train & Render Methods**
**SparseNeRF:**
```bash
python run_nerf.py --config configs/fern_3v.txt \
  --use_dpt_ranking --no_batching --N_iters 1000 --lrate 0.01 \
  --lrate_decay 10 --lambda_rank 0.2 --lambda_cont 0.02 \
  --render_factor 8 --i_video 1000
```
**DSNeRF:**
```bash
python run_nerf.py --config configs/fern_3v_ds.txt \
  --finest_res 1024 --log2_hashmap_size 19 --lrate 0.01 \
  --lrate_decay 10 --render_factor 8 --i_video 1000
  ```
**HashNeRF:**
```bash
python run_nerf.py --config configs/fern_3v_hash.txt \
  --render_factor 8 --i_video 1000
```

> Outputs (videos, depth maps, logs) are saved in logs/ by default. Use the provided compare_all_methods.py script (in scripts/) to automate comparison. 