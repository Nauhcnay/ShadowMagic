## let's train all the models one by one
# anythingV5
export MODEL_DIR="frankjoshua/AnythingV5Ink_ink"
export OUTPUT_DIR="checkpoints/anythingV5"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=512 \
 --num_train_epochs=75\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=ShadowMagicSD1.5\
 --validation_steps=500\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --train_batch_size=4 \
 --report_to="wandb"\
 --checkpointing_steps=500\
 --validation_steps=250

# divineelegancemix
export MODEL_DIR="stablediffusionapi/divineelegancemix"
export OUTPUT_DIR="checkpoints/divineelegancemix"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=512 \
 --num_train_epochs=75\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=ShadowMagicSD1.5\
 --validation_steps=500\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --train_batch_size=4 \
 --report_to="wandb"\
 --checkpointing_steps=500\
 --validation_steps=250

# Hassaku
export MODEL_DIR="dwarfbum/Hassaku"
export OUTPUT_DIR="checkpoints/Hassaku"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=512 \
 --num_train_epochs=75\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=ShadowMagicSD1.5\
 --validation_steps=500\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png"\
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting"\
 --train_batch_size=4 \
 --report_to="wandb"\
 --checkpointing_steps=500\
 --validation_steps=250