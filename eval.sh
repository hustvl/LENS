export PYTHONPATH=$PYTHONPATH:$(pwd)

# stage1 
torchrun --standalone --nproc_per_node 1 eval/evaluate_res.py \
    --model_path './weights/stage1/qwen2p5_87p5' \
    --image_dir './datasets' \
    --if_stage1 


# stage2
# refercoco series
torchrun --standalone --nproc_per_node 1 eval/evaluate_res2.py \
    --model_path './weights/stage2/qwen2p5_refcoco' \
    --image_dir './datasets'

# reasonseg
torchrun --standalone --nproc_per_node 1 eval/evaluate_reasonseg.py \
    --model_path './weights/stage2/qwen2p5_reasonseg' \
    --image_dir './datasets'

# reasonseg cot
torchrun --standalone --nproc_per_node 1 eval/evaluate_reasonseg.py \
    --model_path './weights/stage2/qwen2p5_reasonseg' \
    --image_dir './datasets' \
    --cot

# groundingsuite-eval
torchrun --standalone --nproc_per_node 1 eval/evaluate_res2_gs.py \
    --model_path './weights/stage2/qwen2p5_reasonseg' \
    --image_dir './datasets/coco/unlabeled2017' \
    --anno_path './datasets/GroundingSuite-Eval.jsonl'
