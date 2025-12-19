import os
import requests
import argparse
from tqdm import tqdm
import zipfile

LOGS = {
    "hashnerf": "https://huggingface.co/datasets/artyogan/nerf_project_logs/resolve/main/logs_hash.zip",
    "sparsenerf": "https://huggingface.co/datasets/artyogan/nerf_project_logs/resolve/main/logs_sparse.zip",
    "dsnef": "https://huggingface.co/datasets/artyogan/nerf_project_logs/resolve/main/logs_ds.zip",
    "ddp": "https://huggingface.co/datasets/artyogan/nerf_project_logs/resolve/main/logs_ddp.zip",
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

def download_and_extract(logs_name, save_dir="."):
    if logs_name not in LOGS:
        print(f"'{logs_name}' logs not found. Available: {list(LOGS.keys())}")
        return
    os.makedirs(save_dir, exist_ok=True)
    url = LOGS[logs_name]
    zip_path = os.path.join(save_dir, f"{logs_name}.zip")
    print(f"Downloading {logs_name} logs...")
    download_file(url, zip_path)
    print("Download complete. Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(save_dir))
    print(f"Unzipped to {os.path.join(save_dir)}")

def list_logs():
    """Print available logs"""
    print("Available logs:")
    for i, logs_name in enumerate(LOGS.keys(), 1):
        print(f"  {i}. {logs_name}")
    print(f"\nTotal: {len(LOGS)} logs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download NeRF logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available logs: {', '.join(LOGS.keys())}"
    )
    parser.add_argument("--logs", type=str, help="Dataset name to download (e.g. cowork1, lego)")
    parser.add_argument("--list", action="store_true", help="List all available logs")
    args = parser.parse_args()
    
    if args.list:
        list_logs()
    elif args.logs:
        download_and_extract(args.logs, '.')
    else:
        parser.print_help()
        print("\n" + "="*50)
        list_logs()