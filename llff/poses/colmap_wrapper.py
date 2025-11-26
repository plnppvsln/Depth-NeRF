import os
import sys
import shutil
import subprocess
from glob import glob
from pathlib import Path, PurePosixPath
import argparse

# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")


def _quote_path(path):
	if not path:
		return path
	if " " in path:
		return f"\"{path}\""
	return path


def _external_colmap_glob():
	if os.name == "nt":
		return os.path.join(ROOT_DIR, "external", "colmap", "*", "COLMAP.bat")
	return os.path.join(ROOT_DIR, "external", "colmap", "*", "bin", "colmap")


def resolve_colmap_binary(preferred_binary=None, allow_download=True):
	"""
	Attempt to locate a COLMAP binary. If nothing is found and we're on
	Windows, try to download it via scripts/download_colmap.bat (matching
	the instant-ngp rollout).
	"""
	candidates = []
	for candidate in (preferred_binary, os.environ.get("COLMAP_BINARY")):
		if candidate:
			candidates.append(candidate)

	system_colmap = shutil.which("colmap")
	if system_colmap:
		candidates.append(system_colmap)

	candidates.extend(glob(_external_colmap_glob()))

	for candidate in candidates:
		if candidate and os.path.exists(candidate):
			return candidate

	if allow_download and os.name == "nt":
		download_script = os.path.join(SCRIPTS_FOLDER, "download_colmap.bat")
		if os.path.exists(download_script):
			print("COLMAP not found. Attempting to download COLMAP from the internet.")
			subprocess.check_call(download_script, shell=True)
			return resolve_colmap_binary(preferred_binary, allow_download=False)

	raise FileNotFoundError(
		"Unable to locate COLMAP binary. Install COLMAP, set COLMAP_BINARY, or "
		"place a downloaded copy under external/colmap/."
	)


# def parse_args():
# 	parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

# 	# parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
# 	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
# 	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
# 	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
# 	parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
# 	parser.add_argument("--images", default="images", help="Input path to the images.")
# 	parser.add_argument("--text", default="colmap_text", help="Input path to the colmap text files (set automatically if --run_colmap is used).")
# 	# parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
# 	# parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering).")
# 	# parser.add_argument("--out", default="transforms.json", help="Output JSON file path.")
# 	parser.add_argument("--vocab_path", default="", help="Vocabulary tree path.")
# 	parser.add_argument("--overwrite", action="store_true", help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
# 	# parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.")
# 	args = parser.parse_args()
# 	return args

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type, colmap_binary=None):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    colmap_binary = colmap_binary or resolve_colmap_binary()

    feature_extractor_args = [
        colmap_binary, 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.single_camera', '1',
            # '--SiftExtraction.use_gpu', '0',
    ]
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        colmap_binary, match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    # mapper_args = [
    #     'colmap', 'mapper', 
    #         '--database_path', os.path.join(basedir, 'database.db'), 
    #         '--image_path', os.path.join(basedir, 'images'),
    #         '--output_path', os.path.join(basedir, 'sparse'),
    #         '--Mapper.num_threads', '16',
    #         '--Mapper.init_min_tri_angle', '4',
    # ]
    mapper_args = [
        colmap_binary, 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )
