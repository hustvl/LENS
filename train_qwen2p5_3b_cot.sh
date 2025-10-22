export PYTHONPATH=$PYTHONPATH:$(pwd)
#!/bin/bash
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12345

# MODIFY HERE: please prepare the env related variables
PR1_PATH="./"
CHECKPOINT_PATH="./outputs" # directory to save the checkpoint
RUN_NAME="qwen2p5_stage2_reasonseg_ft_cot" # describe what your experiment is about

# Default Setting
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}" # path to save the output
SRC_PATH="${OUTPUT_DIR}/src" # path to backup the source code

export LOG_DIR="${OUTPUT_DIR}/logs" # path to save the log
export WANDB_PROJECT="LENS" # project name in wandb
export WANDB_TAGS="qwen2p5_stage2_cot" # tags for the experiment in wandb
export WANDB_MODE=offline 

if [ ! -d "${OUTPUT_DIR}"/src ]; then
    mkdir -p ${OUTPUT_DIR}/src
fi

# backup the source code
cp -r ${PR1_PATH}/src ${SRC_PATH}
mkdir -p ${LOG_DIR}

# ReasonSeg Finetune: --question_template "pr1_grounding"
# CoT: --question_template "samr1_v4"
torchrun \
    --nproc_per_node="4" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    ${PR1_PATH}/src/open_r1/grpo_vllm_sam_stage2.py \
    --deepspeed ${PR1_PATH}/configs/zero3.json \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path ./weights/stage2/qwen2p5_8epoch1500step \
    --max_prompt_length 2048 \
    --max_completion_length 768 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --max_pixels 1000000 \
    --num_train_epochs 40 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --learning_rate 3e-6 \
    --reward_funcs "pr1_grounding" "think_format" \
    --save_only_model false \
    --system_prompt_template "default" \
    --question_template "samr1_v4" \
    --train_sample_size 5000000000000 \
    --skip_special_tokens false \
    --answer_template "default" \
    --if_detach_res_loss false \
    --if_use_mask_iou_reward true \
    --if_square_mask_iou_as_reward true \
    --coord_norm_type "qwen2p5vl"
