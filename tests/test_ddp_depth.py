import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
project_root = Path(__file__).parent.parent


def test_dpt_depth(dpt_path="llff_depth_completion"):
    depth_path = os.path.join(project_root, dpt_path, 'depth_000.npy')
    depth_std_path = os.path.join(project_root, dpt_path, 'depth_000_std.npy')

    data = np.load(depth_path)
    data_std = np.load(depth_std_path)

    print("Depth shape:", data.shape)
    print("Std shape:", data_std.shape)

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(data, cmap='magma')
    axes[0].set_title('DPT Depth Map')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(data_std, cmap='magma')
    axes[1].set_title('DPT Depth Std')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_dpt_depth()
