import re
import os
import numpy as np
import warnings
from datetime import datetime
from src.open_r1.utils import extract_bbox_answer, compute_iou

def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_DIR", None)
    os.makedirs(log_dir, exist_ok=True)
    if log_dir is None:
        warnings.warn("LOG_DIR is not set, log will not be saved")
        return
    log_path = os.path.join(log_dir, f"{tag}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
            if other_info is not None:
                for k, v in other_info.items():
                    f.write(f"{k}: {v}\n")
        except:
            f.write("writeing error")

def format_reward(completions, pattern, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, flags=re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def think_format_reward(completions, **kwargs):
    """<think>...</think>\n<answer>...</answer>"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return format_reward(completions, pattern)

def pr1_grounding_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<\|object_ref_start\|>.*?<\|object_ref_end\|><\|box_start\|>.*?<\|box_end\|><\|im_end\|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def pr1_grounding_format_reward_max_0p1(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<|object_ref_start|>.*?<|object_ref_end|><|box_start|>.*?<|box_end|><|im_end|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [0.1 if match else 0.0 for match in matches]

def pr1_grounding_format_reward_reason(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\n<\|object_ref_start\|>.*?<\|object_ref_end\|><\|box_start\|>.*?<\|box_end\|><\|im_end\|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def pr1_grounding_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    for completion, sol in zip(contents, solution):
        bbox = extract_bbox_answer(completion)
        iou = compute_iou(bbox, eval(sol))
        rewards.append(iou**2)
        log(completion + f"\nBounding box: {bbox}", sol, None, iou**2, "pr1_grounding_reward")
    return rewards