import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from load_dataset.llff import load_colmap_depth

basedir = './data/nerf_llff_data/shaving_set_25v/'

def test_load_colmap_depth():
    load_colmap_depth(basedir)

if __name__ == "__main__":
    test_load_colmap_depth()