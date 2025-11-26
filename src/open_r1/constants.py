from src.open_r1.rewards import *

# REWARD MAPING
reward_funcs_registry = {
    "pr1_grounding": pr1_grounding_reward,
    "pr1_grounding_format": pr1_grounding_format_reward,
    "pr1_grounding_format_max_0p1": pr1_grounding_format_reward_max_0p1,
    "pr1_grounding_format_reason": pr1_grounding_format_reward_reason,
    "think_format": think_format_reward,
}

# SYSTEM PROMPTS
LLAVA_SYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

QWEN2_SYS = (
    "You are a helpful assistant. "
)

R1V_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "llava": LLAVA_SYS,
    "qwen": QWEN2_SYS,
    "r1v": R1V_SYS,
}

SAMR1_V4_TEMPLATE = \
    "Locate \"{question}\", report the bbox coordinates in JSON format." \
    "Compare the difference between objects and find the most closely matched one." \
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    "Output the one bbox inside the interested object in JSON format." \
    "i.e., <think>thinking process here</think>" \
    "<answer>answer here</answer>"

question_template_registry = {
    "default": "{question}",
    "pr1_grounding": "Output the bounding box of the {question} in the image.",
    "samr1_v4": SAMR1_V4_TEMPLATE,
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>"
}
