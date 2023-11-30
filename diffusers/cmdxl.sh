## let's train all the models one by one
# anythingV5
# export MODEL_DIR="frankjoshua/AnythingV5Ink_ink"
# export OUTPUT_DIR="checkpoints/anythingV5"
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=waterheater/shadowmagic\
 --resolution=512 \
 --num_train_epochs=100\
 --learning_rate=1e-5 \
 --validation_image "./validation/04462_cond.png" "./validation/03532_cond.png" \
 --validation_prompt "add shadow from right lighting" "add shadow from left lighting" \
 --train_batch_size=4 \
 --report_to="wandb"\
 --checkpointing_steps=500\
 --validation_steps=250

# export MODEL_DIR="stablediffusionapi/divineelegancemix"
# export OUTPUT_DIR="checkpoints/divineelegancemix"
export MODEL_DIR="dwarfbum/Hassaku"
export OUTPUT_DIR="checkpoints/Hassaku"


