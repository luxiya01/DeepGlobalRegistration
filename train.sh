module load Anaconda/2021.05-nsc1
conda activate /proj/berzelius-2023-154/users/x_lling/.conda/envs/dgr-o3d-0.17

python mbes_train.py \
    --mbes_config mbes_data/configs/mbes_crop_train.yaml \
    --network_config network_configs/mbes-train.yaml