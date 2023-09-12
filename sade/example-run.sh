CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python main.py \
    --project test --mode train \
    --config configs/ve/toy_config.py --workdir workdir/test/ \
    --config.data.num_channels=2 \
    --config.data.dir_path /DATA/Users/amahmood/braintyp/processed_v2/ \
    --config.data.splits_dir /codespace/sade/sade/datasets/brains/ \
    --config.data.ood_ds lesion \

