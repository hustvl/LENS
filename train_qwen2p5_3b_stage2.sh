export PYTHONPATH=$PYTHONPATH:$(pwd)
#!/bin/bash
NNODES=4
NODE_RANK=0
MASTER_ADDR=Master_IP  # ! MODIFY HERE
MASTER_PORT=12345

# MODIFY HERE: please prepare the env related variables
PR1_PATH="./"
CHECKPOINT_PATH="./outputs" # directory to save the checkpoint
RUN_NAME="qwen2p5_stage2"

# Default Setting
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}" # path to save the output
SRC_PATH="${OUTPUT_DIR}/src" # path to backup the source code

export LOG_DIR="${OUTPUT_DIR}/logs" # path to save the log
export WANDB_PROJECT="LENS" # project name in wandb
export WANDB_TAGS="qwen2p5_stage2" # tags for the experiment in wandb
export WANDB_MODE=offline 

if [ ! -d "${OUTPUT_DIR}"/src ]; then
    mkdir -p ${OUTPUT_DIR}/src
fi

# backup the source code
cp -r ${PR1_PATH}/src ${SRC_PATH}
mkdir -p ${LOG_DIR}

# run the training
torchrun \
    --nproc_per_node="4" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${PR1_PATH}/src/open_r1/grpo_vllm_sam_stage2.py \
    --deepspeed ${PR1_PATH}/configs/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path ./weights/stage1/qwen2p5_87p5 \
    --max_prompt_length 2048 \
    --max_completion_length 768 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 16 \
    --num_generations 8 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 8 \
    --run_name ${RUN_NAME} \
    --save_strategy "steps" \
    --save_steps 500 \
    --reward_funcs "pr1_grounding" \
    --save_only_model false \
    --system_prompt_template "default" \
    --question_template "pr1_grounding" \
    --train_sample_size 5000000000000 \
    --skip_special_tokens false \
    --answer_template "default" \
    --learning_rate 3e-6 \
    --if_detach_res_loss false \
    --if_use_mask_iou_reward true \
    --if_square_mask_iou_as_reward true \
    --coord_norm_type "qwen2p5vl"
