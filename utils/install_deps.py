"""
Install compatible PyTorch and dependencies based on available hardware.
Supports Windows, Linux, macOS.
"""
import subprocess
import sys
import platform
import os
import re
import json
from typing import Optional, Tuple

def run(cmd):
    """Execute pip command with proper error handling."""
    pip_cmd = [sys.executable, "-m", "pip"] + cmd
    print(f" Running: {' '.join(pip_cmd)}")
    try:
        subprocess.check_call(pip_cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f" Error executing command: {e}")
        return False
    except Exception as e:
        print(f" Unexpected error: {e}")
        return False

def get_cuda_version() -> Optional[str]:
    """
    Detect installed CUDA version by multiple methods.
    Returns version string like '12.1' or None if not found.
    """
    # Method 1: Try to get CUDA version from nvcc
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout
            # Look for version pattern like "release 12.1" or "V11.8.0"
            match = re.search(r'release\s+(\d+\.\d+)|V(\d+\.\d+)\.\d+', output)
            if match:
                version = match.group(1) or match.group(2)
                print(f" Detected CUDA version from nvcc: {version}")
                return version
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        pass  # nvcc not found or other error

    # Method 2: Check environment variables
    cuda_paths = []
    if "CUDA_HOME" in os.environ:
        cuda_paths.append(os.environ["CUDA_HOME"])
    if "CUDA_PATH" in os.environ:
        cuda_paths.append(os.environ["CUDA_PATH"])
    
    # Common default paths
    if platform.system() == "Linux":
        cuda_paths.extend(["/usr/local/cuda", "/usr/cuda"])
    elif platform.system() == "Windows":
        cuda_paths.extend([os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA GPU Computing Toolkit\\CUDA"),
                          os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "NVIDIA GPU Computing Toolkit\\CUDA")])
    elif platform.system() == "Darwin":  # macOS
        cuda_paths.extend(["/usr/local/cuda", "/Developer/NVIDIA/CUDA"])

    # Check version.txt files
    for path in cuda_paths:
        if not os.path.exists(path):
            continue
            
        # Look for version.txt in CUDA directory
        version_file = os.path.join(path, "version.txt")
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    content = f.read()
                    match = re.search(r'(\d+\.\d+)', content)
                    if match:
                        version = match.group(1)
                        print(f" Detected CUDA version from version.txt: {version}")
                        return version
            except Exception as e:
                continue
        
        # Look for version.json in newer CUDA installations
        version_json = os.path.join(path, "version.json")
        if os.path.exists(version_json):
            try:
                with open(version_json, 'r') as f:
                    data = json.load(f)
                    version = data.get('cuda_nvcc_version', '').split('.')[0] + '.' + data.get('cuda_nvcc_version', '').split('.')[1]
                    if version:
                        print(f" Detected CUDA version from version.json: {version}")
                        return version
            except Exception as e:
                continue
        
        # Check include/cuda.h for version macros
        cuda_h = os.path.join(path, "include", "cuda.h")
        if os.path.exists(cuda_h):
            try:
                with open(cuda_h, 'r') as f:
                    content = f.read()
                    # Look for CUDA_VERSION macro
                    match = re.search(r'#define\s+CUDA_VERSION\s+(\d+)', content)
                    if match:
                        version_num = int(match.group(1))
                        major = version_num // 1000
                        minor = (version_num % 1000) // 10
                        version = f"{major}.{minor}"
                        print(f" Detected CUDA version from cuda.h: {version}")
                        return version
            except Exception as e:
                continue

    # Method 3: Check nvidia-smi output for driver-supported CUDA version
    try:
        result = subprocess.run(["nvidia-smi", "--query", "gpu.Driver Version", "--format=csv"], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            driver_version = result.stdout.strip().split('\n')[1]
            print(f" NVIDIA driver version: {driver_version}")
            # Note: This shows max supported CUDA version by driver, not installed toolkit version
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        pass

    print(" Could not determine exact CUDA toolkit version. Will try to use latest supported version.")
    return None

def get_supported_cuda_version(detected_version: Optional[str]) -> str:
    """
    Map detected CUDA version to supported PyTorch CUDA version.
    Returns '12.1', '11.8', or 'cpu' as fallback.
    """
    supported_versions = ['12.1', '11.8']
    
    if detected_version is None:
        print(" No CUDA version detected. Checking GPU availability...")
        return None
    
    # Extract major and minor version
    try:
        major, minor = map(int, detected_version.split('.')[:2])
        detected_major_minor = f"{major}.{minor}"
        
        # Check if detected version matches supported versions exactly
        if detected_major_minor in supported_versions:
            return detected_major_minor
        
        # Check for close matches (e.g., 12.2 -> use 12.1, 11.7 -> use 11.8)
        detected_float = float(detected_major_minor)
        for supported in sorted([float(v) for v in supported_versions], reverse=True):
            if detected_float >= supported - 0.2:  # Allow some flexibility
                return f"{int(supported)}.{int((supported - int(supported)) * 10)}"
        
    except (ValueError, TypeError) as e:
        print(f" Error parsing CUDA version: {e}")
    
    return None

def check_cuda_available() -> bool:
    """Check if CUDA is available on the system."""
    try:
        # First try to import torch (might be already installed)
        import torch
        if torch.cuda.is_available():
            print(" CUDA available via torch.cuda.is_available()")
            return True
    except ImportError:
        pass  # torch not installed yet
    
    # Check if nvidia-smi is available (indicates NVIDIA drivers are installed)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(" NVIDIA GPU detected via nvidia-smi")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        pass
    
    # Check for CUDA environment variables
    if "CUDA_HOME" in os.environ or "CUDA_PATH" in os.environ:
        print(" CUDA environment variables found")
        return True
    
    print(" No CUDA detected")
    return False

def main():
    # Устанавливаем не-PyTorch зависимости из requirements-base.txt
    print("Installing base dependencies from requirements-base.txt...")
    if not os.path.exists("requirements-base.txt"):
        print(" Warning: requirements-base.txt not found. Creating minimal version...")
        with open("requirements-base.txt", "w") as f:
            f.write("numpy\n")
            f.write("pillow\n")
            f.write("requests\n")
    
    success = run(["install", "-r", "requirements-base.txt"])
    if not success:
        print(" Warning: Failed to install base dependencies, continuing anyway...")

    # Определяем, есть ли CUDA
    cuda_available = check_cuda_available()
    
    torch_cmd = None
    
    if cuda_available:
        # Получаем последнюю совместимую версию CUDA
        detected_cuda_version = get_cuda_version()
        supported_cuda_version = get_supported_cuda_version(detected_cuda_version)
        
        if supported_cuda_version == '12.1':
            print(" Using PyTorch with CUDA 12.1")
            torch_cmd = ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
        elif supported_cuda_version == '11.8':
            print(" Using PyTorch with CUDA 11.8")
            torch_cmd = ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
        else:
            # Fallback to CUDA 12.1 if we can't determine exact version but GPU is available
            print(" Could not determine specific CUDA version. Using latest supported CUDA 12.1")
            torch_cmd = ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
    else:
        # CPU-only
        print(" No CUDA detected. Using CPU-only PyTorch")
        torch_cmd = ["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]

    # Устанавливаем PyTorch
    success = run(torch_cmd)
    if not success:
        print(" Warning: Failed to install PyTorch. Trying alternative approach...")
        # Try without index-url as fallback
        fallback_cmd = ["install", "torch", "torchvision", "torchaudio"]
        run(fallback_cmd)

    # Verify installation
    try:
        import torch
        print(f" PyTorch version: {torch.__version__}")
        print(f" CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" CUDA version: {torch.version.cuda}")
            print(f" GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print(" Error: PyTorch was not installed successfully")

    print("\n All dependencies installation completed!")

if __name__ == "__main__":
    print("PyTorch Installation Script")
    print("==========================")
    print(f" Python version: {platform.python_version()}")
    print(f" Platform: {platform.platform()}")
    print(f" System: {platform.system()}")
    print()
    
    main()