CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python main.py \
    --project test --mode train \
    --config configs/ve/toy_config.py --workdir workdir/test/