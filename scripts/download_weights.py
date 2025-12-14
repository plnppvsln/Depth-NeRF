import os
import sys
import urllib.request

# Add the project root to the Python path to ensure weights are saved correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

DPT_URL = "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
OUTPUT_PATH = os.path.join(WEIGHTS_DIR, "dpt_hybrid-midas-501f0c75.pt")

if os.path.exists(OUTPUT_PATH):
    print("DPT weights already exist.")
else:
    print("Downloading DPT weights...")
    urllib.request.urlretrieve(DPT_URL, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")