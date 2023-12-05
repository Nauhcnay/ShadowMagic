## let's train all the models one by one
# anythingV5
export MODEL_DIR="frankjoshua/AnythingV5Ink_ink"
export OUTPUT_DIR="checkpoints/anythingv5_2x"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagicxl\
 --resolution=1024 \
 --num_train_epochs=450\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=ShadowMagicSD1.5\
 --validation_steps=1000\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sdxl/image11_color.png" "./validation/sdxl/image11_line.png" "./validation/sdxl/image12_color.png" "./validation/sdxl/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --train_batch_size=2 \
 --report_to="wandb"\
 --checkpointing_steps=1000\
 --validation_steps=1000\
 --gradient_accumulation_steps=4

# divineelegancemix
export MODEL_DIR="stablediffusionapi/divineelegancemix"
export OUTPUT_DIR="checkpoints/divineelegancemix_2x"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=1024 \
 --num_train_epochs=450\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=ShadowMagicSD1.5\
 --validation_steps=1000\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sdxl/image11_color.png" "./validation/sdxl/image11_line.png" "./validation/sdxl/image12_color.png" "./validation/sdxl/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --train_batch_size=2 \
 --report_to="wandb"\
 --checkpointing_steps=1000\
 --validation_steps=1000\
 --gradient_accumulation_steps=4

# MeinaPastel
export MODEL_DIR="Meina/MeinaPastel_V6"
export OUTPUT_DIR="checkpoints/MeinaPastel_2x"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=1024 \
 --num_train_epochs=450\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=ShadowMagicSD1.5\
 --validation_steps=1000\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sdxl/image11_color.png" "./validation/sdxl/image11_line.png" "./validation/sdxl/image12_color.png" "./validation/sdxl/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --train_batch_size=2 \
 --report_to="wandb"\
 --checkpointing_steps=1000\
 --validation_steps=1000\
 --gradient_accumulation_steps=4