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
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png" "./validation/sd1.5/image143_color.png" "./validation/sd1.5/image143_line.png" "./validation/sd1.5/image149_color.png" "./validation/sd1.5/image149_line.png" "./validation/sd1.5/image164_color.png" "./validation/sd1.5/image164_line.png" "./validation/sd1.5/image193_color.png" "./validation/sd1.5/image193_line.png" "./validation/sd1.5/image197_color.png" "./validation/sd1.5/image197_line.png" "./validation/sd1.5/image204_color.png" "./validation/sd1.5/image204_line.png" "./validation/sd1.5/image228_color.png" "./validation/sd1.5/image228_line.png" "./validation/sd1.5/image23_color.png" "./validation/sd1.5/image23_line.png" "./validation/sd1.5/image24_color.png" "./validation/sd1.5/image24_line.png" "./validation/sd1.5/image258_color.png" "./validation/sd1.5/image258_line.png" "./validation/sd1.5/image279_color.png" "./validation/sd1.5/image279_line.png" "./validation/sd1.5/image298_color.png" "./validation/sd1.5/image298_line.png" "./validation/sd1.5/image341_color.png" "./validation/sd1.5/image341_line.png" "./validation/sd1.5/image363_color.png" "./validation/sd1.5/image363_line.png" "./validation/sd1.5/image364_color.png" "./validation/sd1.5/image364_line.png" "./validation/sd1.5/image37_color.png" "./validation/sd1.5/image37_line.png" "./validation/sd1.5/image59_color.png" "./validation/sd1.5/image59_line.png" "./validation/sd1.5/image7_color.png" "./validation/sd1.5/image7_line.png" \
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from back lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" \
 --validation_steps=500 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="wandb" \
 --seed=42 \