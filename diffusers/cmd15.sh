## let's train all the models one by one
# anythingV5
export MODEL_DIR="frankjoshua/AnythingV5Ink_ink"
export OUTPUT_DIR="checkpoints/anythingV5"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=512 \
 --num_train_epochs=100\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=AnythingV5\
 --validation_steps=500\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png" "./validation/sd1.5/image143_color.png" "./validation/sd1.5/image143_line.png" "./validation/sd1.5/image149_color.png" "./validation/sd1.5/image149_line.png" "./validation/sd1.5/image164_color.png" "./validation/sd1.5/image164_line.png" "./validation/sd1.5/image193_color.png" "./validation/sd1.5/image193_line.png" "./validation/sd1.5/image197_color.png" "./validation/sd1.5/image197_line.png" "./validation/sd1.5/image204_color.png" "./validation/sd1.5/image204_line.png" "./validation/sd1.5/image228_color.png" "./validation/sd1.5/image228_line.png" "./validation/sd1.5/image23_color.png" "./validation/sd1.5/image23_line.png" "./validation/sd1.5/image24_color.png" "./validation/sd1.5/image24_line.png" "./validation/sd1.5/image258_color.png" "./validation/sd1.5/image258_line.png" "./validation/sd1.5/image279_color.png" "./validation/sd1.5/image279_line.png" "./validation/sd1.5/image298_color.png" "./validation/sd1.5/image298_line.png" "./validation/sd1.5/image341_color.png" "./validation/sd1.5/image341_line.png" "./validation/sd1.5/image363_color.png" "./validation/sd1.5/image363_line.png" "./validation/sd1.5/image364_color.png" "./validation/sd1.5/image364_line.png" "./validation/sd1.5/image37_color.png" "./validation/sd1.5/image37_line.png" "./validation/sd1.5/image59_color.png" "./validation/sd1.5/image59_line.png" "./validation/sd1.5/image7_color.png" "./validation/sd1.5/image7_line.png" \
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from back lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" \
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
 --num_train_epochs=100\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=DivineeleganceMix\
 --validation_steps=500\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png" "./validation/sd1.5/image143_color.png" "./validation/sd1.5/image143_line.png" "./validation/sd1.5/image149_color.png" "./validation/sd1.5/image149_line.png" "./validation/sd1.5/image164_color.png" "./validation/sd1.5/image164_line.png" "./validation/sd1.5/image193_color.png" "./validation/sd1.5/image193_line.png" "./validation/sd1.5/image197_color.png" "./validation/sd1.5/image197_line.png" "./validation/sd1.5/image204_color.png" "./validation/sd1.5/image204_line.png" "./validation/sd1.5/image228_color.png" "./validation/sd1.5/image228_line.png" "./validation/sd1.5/image23_color.png" "./validation/sd1.5/image23_line.png" "./validation/sd1.5/image24_color.png" "./validation/sd1.5/image24_line.png" "./validation/sd1.5/image258_color.png" "./validation/sd1.5/image258_line.png" "./validation/sd1.5/image279_color.png" "./validation/sd1.5/image279_line.png" "./validation/sd1.5/image298_color.png" "./validation/sd1.5/image298_line.png" "./validation/sd1.5/image341_color.png" "./validation/sd1.5/image341_line.png" "./validation/sd1.5/image363_color.png" "./validation/sd1.5/image363_line.png" "./validation/sd1.5/image364_color.png" "./validation/sd1.5/image364_line.png" "./validation/sd1.5/image37_color.png" "./validation/sd1.5/image37_line.png" "./validation/sd1.5/image59_color.png" "./validation/sd1.5/image59_line.png" "./validation/sd1.5/image7_color.png" "./validation/sd1.5/image7_line.png" \
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from back lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" \
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
 --num_train_epochs=100\
 --learning_rate=1e-5 \
 --checkpoints_total_limit=20\
 --tracker_project_name=Hassaku\
 --validation_steps=500\
 --cache_dir=dataset/sd15\
 --validation_image "./validation/sd1.5/image11_color.png" "./validation/sd1.5/image11_line.png" "./validation/sd1.5/image12_color.png" "./validation/sd1.5/image12_line.png" "./validation/sd1.5/image143_color.png" "./validation/sd1.5/image143_line.png" "./validation/sd1.5/image149_color.png" "./validation/sd1.5/image149_line.png" "./validation/sd1.5/image164_color.png" "./validation/sd1.5/image164_line.png" "./validation/sd1.5/image193_color.png" "./validation/sd1.5/image193_line.png" "./validation/sd1.5/image197_color.png" "./validation/sd1.5/image197_line.png" "./validation/sd1.5/image204_color.png" "./validation/sd1.5/image204_line.png" "./validation/sd1.5/image228_color.png" "./validation/sd1.5/image228_line.png" "./validation/sd1.5/image23_color.png" "./validation/sd1.5/image23_line.png" "./validation/sd1.5/image24_color.png" "./validation/sd1.5/image24_line.png" "./validation/sd1.5/image258_color.png" "./validation/sd1.5/image258_line.png" "./validation/sd1.5/image279_color.png" "./validation/sd1.5/image279_line.png" "./validation/sd1.5/image298_color.png" "./validation/sd1.5/image298_line.png" "./validation/sd1.5/image341_color.png" "./validation/sd1.5/image341_line.png" "./validation/sd1.5/image363_color.png" "./validation/sd1.5/image363_line.png" "./validation/sd1.5/image364_color.png" "./validation/sd1.5/image364_line.png" "./validation/sd1.5/image37_color.png" "./validation/sd1.5/image37_line.png" "./validation/sd1.5/image59_color.png" "./validation/sd1.5/image59_line.png" "./validation/sd1.5/image7_color.png" "./validation/sd1.5/image7_line.png" \
 --validation_prompt "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from left lighting and remove color" "add shadow from left lighting" "add shadow from left lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from right lighting" "add shadow from right lighting and remove color" "add shadow from left lighting" "add shadow from right lighting and remove color" "add shadow from top lighting" "add shadow from top lighting and remove color" "add shadow from left lighting" "add shadow from back lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" "add shadow from left lighting and remove color" "add shadow from right lighting" \
 --train_batch_size=4 \
 --report_to="wandb"\
 --checkpointing_steps=500\
 --validation_steps=250