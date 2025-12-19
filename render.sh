python run_nerf.py --config configs/sparsenerf.txt --expname fern_2v_sparse --datadir ./data/nerf_llff_data/fern_2v --render_only --render_factor 2
python run_nerf.py --config configs/sparsenerf.txt --expname fern_3v_sparse --datadir ./data/nerf_llff_data/fern_3v --render_only --render_factor 2
python run_nerf.py --config configs/sparsenerf.txt --expname fern_5v_sparse --datadir ./data/nerf_llff_data/fern_5v --render_only --render_factor 2
python run_nerf.py --config configs/sparsenerf.txt --expname fern_9v_sparse --datadir ./data/nerf_llff_data/fern_9v --render_only --render_factor 2

python run_nerf.py --config configs/dsnerf.txt --expname fern_2v_ds --datadir ./data/nerf_llff_data/fern_2v --render_only --render_factor 2
python run_nerf.py --config configs/dsnerf.txt --expname fern_3v_ds --datadir ./data/nerf_llff_data/fern_3v --render_only --render_factor 2
python run_nerf.py --config configs/dsnerf.txt --expname fern_5v_ds --datadir ./data/nerf_llff_data/fern_5v --render_only --render_factor 2
python run_nerf.py --config configs/dsnerf.txt --expname fern_9v_ds --datadir ./data/nerf_llff_data/fern_9v --render_only --render_factor 2

python run_nerf.py --config configs/hashnerf.txt --expname fern_2v_hash --datadir ./data/nerf_llff_data/fern_2v --render_only --render_factor 2
python run_nerf.py --config configs/hashnerf.txt --expname fern_3v_hash --datadir ./data/nerf_llff_data/fern_3v --render_only --render_factor 2
python run_nerf.py --config configs/hashnerf.txt --expname fern_5v_hash --datadir ./data/nerf_llff_data/fern_5v --render_only --render_factor 2
python run_nerf.py --config configs/hashnerf.txt --expname fern_9v_hash --datadir ./data/nerf_llff_data/fern_9v --render_only --render_factor 2

python run_nerf.py --config configs/ddp.txt --expname fern_2v_ddp --datadir ./data/nerf_llff_data/fern_2v --render_only --render_factor 2
python run_nerf.py --config configs/ddp.txt --expname fern_3v_ddp --datadir ./data/nerf_llff_data/fern_3v --render_only --render_factor 2
python run_nerf.py --config configs/ddp.txt --expname fern_5v_ddp --datadir ./data/nerf_llff_data/fern_5v --render_only --render_factor 2
python run_nerf.py --config configs/ddp.txt --expname fern_9v_ddp --datadir ./data/nerf_llff_data/fern_9v --render_only --render_factor 2