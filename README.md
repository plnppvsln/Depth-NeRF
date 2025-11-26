# Depth-NeRF

## Downloading Datasets

The project includes a convenient script to download pre-processed NeRF datasets.

### List Available Datasets

To see all available datasets:

```bash
python download_dataset.py --list
```

This will display all available datasets that can be downloaded.

### Download a Dataset

To download a specific dataset:

```bash
python download_dataset.py --dataset <dataset_name>
```

For example:

```bash
python download_dataset.py --dataset lego
python download_dataset.py --dataset fox
python download_dataset.py --dataset shaving_set
python download_dataset.py --dataset fern
```

### Available Datasets

- `shaving_set`
- `lego`
- `fern`
- `fox`

### Custom Save Directory

By default, datasets are saved to the `./data` directory. You can specify a custom directory:

```bash
python download_dataset.py --dataset lego --save_dir /path/to/custom/directory
```

**Note:** The default save directory (`./data`) is recommended as it matches the expected project structure.

### Help

For more information, run:

```bash
python download_dataset.py --help
```

## Running COLMAP

COLMAP is used to estimate camera poses from a set of images. This is necessary for custom datasets or when poses are not pre-computed.

### Prerequisites

COLMAP will be automatically downloaded on Windows if not found. On Linux/Mac, you can either:
- Install COLMAP system-wide: `sudo apt-get install colmap` (Linux) or `brew install colmap` (Mac)
- Set the `COLMAP_BINARY` environment variable to point to your COLMAP installation
- Place a COLMAP binary in `external/colmap/`

### Running COLMAP on a Scene

To process images and generate camera poses, use the `imgs2poses.py` script:

```bash
python imgs2poses.py <scene_directory>
```

For example:

```bash
python imgs2poses.py data/nerf_llff_data/fern
python imgs2poses.py data/nerf_custom/fox
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

### Match Types

You can specify the matching algorithm using the `--match_type` parameter:

- `exhaustive_matcher` (default): Best for unordered image collections
- `sequential_matcher`: Best for video sequences or ordered images

```bash
python imgs2poses.py --match_type sequential_matcher <scene_directory>
```

### Output

After running COLMAP, the following files will be created in your scene directory:

- `database.db`: COLMAP feature database
- `sparse/0/`: Sparse reconstruction containing camera poses
  - `cameras.bin`: Camera parameters
  - `images.bin`: Image poses and metadata
  - `points3D.bin`: 3D point cloud
- `poses_bounds.npy`: Processed poses in NeRF format
- `colmap_output.txt`: COLMAP processing logs

The script will automatically skip COLMAP if the sparse reconstruction already exists.