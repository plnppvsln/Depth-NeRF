import os
import requests
import argparse
from tqdm import tqdm
import zipfile

DATASETS = {
    # "shaving_set": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/shaving_set.zip",
    # "lego": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/lego.zip",
    "fern": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/fern.zip",
    # "fox": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/fox.zip",
    "cowork1": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/cowork1.zip",
    "cowork2": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/cowork2.zip",
    # "table": "https://huggingface.co/datasets/artyogan/nerf_project_datasets/resolve/main/table.zip",
}

def download_file(url, dest):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as file, tqdm(
        desc=dest,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_and_extract(dataset_name, save_dir="data"):
    if dataset_name not in DATASETS:
        print(f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")
        return
    os.makedirs(save_dir, exist_ok=True)
    url = DATASETS[dataset_name]
    zip_path = os.path.join(save_dir, f"{dataset_name}.zip")
    print(f"Downloading {dataset_name} dataset...")
    download_file(url, zip_path)
    print("Download complete. Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(save_dir))
    print(f"Unzipped to {os.path.join(save_dir)}")

def list_datasets():
    """Print available datasets"""
    print("Available datasets:")
    for i, dataset_name in enumerate(DATASETS.keys(), 1):
        print(f"  {i}. {dataset_name}")
    print(f"\nTotal: {len(DATASETS)} datasets")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download NeRF datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(DATASETS.keys())}"
    )
    parser.add_argument("--dataset", type=str, help="Dataset name to download (e.g. cowork1, lego)")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save datasets (by default is ./data and should be it)")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
    elif args.dataset:
        download_and_extract(args.dataset, args.save_dir)
    else:
        parser.print_help()
        print("\n" + "="*50)
        list_datasets()