export MODEL_DIR="Linaqruf/animagine-xl-2.0"
export OUTPUT_DIR="checkpoints/sdxl"

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagicxl\
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --tracker_project_name=ShadowMagicSDXL\
 --cache_dir=dataset/sdxl\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --validation_steps=250 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \