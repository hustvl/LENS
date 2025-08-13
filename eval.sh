export PYTHONPATH=$PYTHONPATH:$(pwd)

# stage1 
torchrun --standalone --nproc_per_node 1 eval/evaluate_res.py \
    --model_path '/path/to/model' \
    --image_dir '/path/to/dir' \
    --if_stage1 


# stage2
# refercoco series
torchrun --standalone --nproc_per_node 1 eval/evaluate_res2.py \
    --model_path '/path/to/model' \
    --image_dir '/path/to/dir'

# reasonseg
torchrun --standalone --nproc_per_node 1 eval/evaluate_reasonseg.py \
    --model_path '/path/to/model' \
    --image_dir '/path/to/dir'

# groundingsuite-eval
torchrun --standalone --nproc_per_node 1 eval/evaluate_res2_gs.py \
    --model_path '/path/to/model' \
    --image_dir '/path/to/coco/unlabeled2017' \
    --anno_path "/path/to/GroundingSuite-Eval.jsonl"
