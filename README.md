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