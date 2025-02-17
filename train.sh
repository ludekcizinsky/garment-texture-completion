MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
VAE_PATH="checkpoints"
DATASET_PATH="datasets/fabric_w_logo"
MODEL_ID="pbr_texture_completion_reproduce"

accelerate launch train.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_path=$DATASET_PATH \
    --vae_path=$VAE_PATH \
    --output_dir=checkpoints/$MODEL_ID \
    --resolution=512 \
    --validation_epochs=1 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=28000 \
    --checkpointing_steps=4000 \
    --checkpoints_total_limit=5 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --ddim_loss \