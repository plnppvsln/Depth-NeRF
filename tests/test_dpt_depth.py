import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
project_root = Path(__file__).parent.parent


def test_dpt_depth(dpt_path="data/nerf_llff_data/fern_5v/dpt_depths"):
    dir = os.path.join(project_root, dpt_path, '001.npy')
    data = np.load(dir)
    print(data.shape)
    # Визуализация глубины
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='viridis')  # или 'plasma', 'inferno' для глубины
    plt.colorbar(label='Inverse Depth')
    plt.title('DPT Depth Map')
    plt.axis('off')
    plt.show()
    return 

if __name__ == "__main__":
    test_dpt_depth()
