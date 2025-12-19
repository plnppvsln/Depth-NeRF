# This one is only for one dataset (cowork2)
python run_nerf.py --config configs/hashnerf.txt --expname cowork2_2v_hash --datadir ./data/nerf_llff_data/cowork2_2v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/hashnerf.txt --expname cowork2_3v_hash --datadir ./data/nerf_llff_data/cowork2_3v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/hashnerf.txt --expname cowork2_5v_hash --datadir ./data/nerf_llff_data/cowork2_5v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/hashnerf.txt --expname cowork2_9v_hash --datadir ./data/nerf_llff_data/cowork2_9v --render_only --render_factor 7 --render_test --eval

python run_nerf.py --config configs/dsnerf.txt --expname cowork2_2v_ds --datadir ./data/nerf_llff_data/cowork2_2v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/dsnerf.txt --expname cowork2_3v_ds --datadir ./data/nerf_llff_data/cowork2_3v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/dsnerf.txt --expname cowork2_5v_ds --datadir ./data/nerf_llff_data/cowork2_5v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/dsnerf.txt --expname cowork2_9v_ds --datadir ./data/nerf_llff_data/cowork2_9v --render_only --render_factor 7 --render_test --eval

python run_nerf.py --config configs/sparsenerf.txt --expname cowork2_2v_sparse --datadir ./data/nerf_llff_data/cowork2_2v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/sparsenerf.txt --expname cowork2_3v_sparse --datadir ./data/nerf_llff_data/cowork2_3v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/sparsenerf.txt --expname cowork2_5v_sparse --datadir ./data/nerf_llff_data/cowork2_5v --render_only --render_factor 7 --render_test --eval
python run_nerf.py --config configs/sparsenerf.txt --expname cowork2_9v_sparse --datadir ./data/nerf_llff_data/cowork2_9v --render_only --render_factor 7 --render_test --eval

python run_nerf.py --config configs/ddp.txt --expname cowork2_2v_ddp --datadir ./data/nerf_llff_data/cowork2_2v --render_only --render_factor 7 --render_test --eval --multi_gpu
python run_nerf.py --config configs/ddp.txt --expname cowork2_3v_ddp --datadir ./data/nerf_llff_data/cowork2_3v --render_only --render_factor 7 --render_test --eval --multi_gpu
python run_nerf.py --config configs/ddp.txt --expname cowork2_5v_ddp --datadir ./data/nerf_llff_data/cowork2_5v --render_only --render_factor 7 --render_test --eval --multi_gpu
python run_nerf.py --config configs/ddp.txt --expname cowork2_9v_ddp --datadir ./data/nerf_llff_data/cowork2_9v --render_only --render_factor 7 --render_test --eval --multi_gpu