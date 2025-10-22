from trl import ScriptArguments
from dataclasses import dataclass, field
from typing import Optional, Union

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    system_prompt_template: Optional[str] = field(
        default="reasoning",
        metadata={"help": "System prompt template. Possible values: 'llava', 'qwen2', 'reasoning', 'grounding', 'ocr'"},
    )
    question_template: Optional[str] = field(
        default="default",
        metadata={"help": "Question template. Possible values: 'default', 'llava', 'qwen2', 'reasoning'"},
    )
    answer_template: Optional[str] = field(
        default="default",
        metadata={"help": "Answer template. Possible values: 'default', 'llava', 'qwen2', 'reasoning'"},
    )
    train_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "Train sample size. If None, use all samples."},
    )
    use_kl: bool = field(
        default=True, 
        metadata={"help":"whether to use kl in loss. If false, no kl will be included into loss. But you can also view kl change trends in pandb"}
    )
    kl_approximator: str = field(
        default="k3", 
        metadata={"help": "which type kl to use for computing loss.you can use k1(not good), k3(official in grpo, unbias, lowest variance), kimikl(only the kl used in kimi1.5), kimifull(the same setting as the core idea of kimi1.5, your value of sync_ref_model, ref_model_mixup_alpha and ref_model_sync_steps will be invalid, they are all set the same as kimi1.5)"}
    )
    reward_scale: float = field(
        default=1, 
        metadata={"help": "reward scale of all rewards"}
    )
    entropy_reg : bool = field(
        default=False, 
        metadata={"help": "whether to use entropy regularization while training. For discriminative tasks like grounding, ocr and counting, we expect entropy to decrease. For literary creation task, we expect entropy to increase. this can be controlled by entropy_weight."}
    )
    entropy_weight: float = field(
        default=0.01, 
        metadata={"help": "the weight for entropy loss. It's only valid when entropy_reg is true. If it's positive, the entropy is to increase. If it's negetive, the entropy is to decrease."}
    )
    temperature_func: str = field(
        default="constant",
        metadata={"help":"which temperature function to use while training. Unlike reward_funcs, you can only use one temperature function."}
    )
    temperature_begin: float = field(
        default=0.1, 
        metadata={"help": "the beginning temperature for training(optional for linear temperature)"}
    )
    temperature_end: float = field(
        default=1.0, 
        metadata={"help": "the ending temperature for training(optional for linear temperature)"}
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "whether to skip special tokens, use when rec task."}
    )
    no_mean_for_same_reward: bool = field(
        default=False,
        metadata={"help": "whether to not minus reward mean if same reward"}
    )
    problem_key: str = field(
        default="problem",
        metadata={"help": "the key of problem in the dataset"}
    )
    answer_key: str = field(
        default="solution",
        metadata={"help": "the key of answer in the dataset"}
    )
    image_key: str = field(
        default="image",
        metadata={"help": "the key of image in the dataset"}
    )
    image_dir: str = field(
        default="",
        metadata={"help": "the directory of image in the dataset"}
    )
    rec_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "the ratio of rec loss to the total loss"}
    )
    res_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "the ratio of res loss to the total loss"}
    )
    if_detach_res_loss: bool = field(
        default=False,
        metadata={"help": "whether to detach res loss"}
    )
    if_freeze_llm: bool = field(
        default=False,
        metadata={"help": "whether to freeze llm"}
    )
    num_of_query: int = field(
        default=64,
        metadata={"help": "the number of query"}
    )
    if_use_qwen_connector: bool = field(
        default=True,
        metadata={"help": "whether to use qwen connector"}
    )
    if_include_sam: bool = field(
        default=False,
        metadata={"help": "whether to include sam"}
    )
    if_use_mask_iou_reward: bool = field(
        default=False,
        metadata={"help": "whether to use mask iou reward"}
    )
    if_use_boundary_iou_reward: bool = field(
        default=False,
        metadata={"help": "whether to use boundary iou reward"}
    )
    if_use_cl: bool = field(
        default=False,
        metadata={"help": "whether to use curriculum learning"}
    )
    if_square_mask_iou_as_reward: bool = field(
        default=False,
        metadata={"help": "whether to square mask iou as reward"}
    )
    if_square_boundary_iou_as_reward: bool = field(
        default=False,
        metadata={"help": "whether to square mask iou as reward"}
    )
    if_use_pixel_reward: bool = field(
        default=False,
        metadata={"help": "whether to use pixel reward"}
    )
    refer_seg_ds: str = field(
        default="refcoco,refcoco+,refcocog",
        metadata={"help": "the dataset to use for referring expression segmentation"}
    )
    coord_norm_type: str = field(
        default="qwen2vl",
        metadata={"help": "the type of coordinate normalization",
                  "choices": "qwen2vl, qwen2p5vl"}
    )
    # str or bool
    resume_checkpoint: str = field(
        default="",
        metadata={"help": "the checkpoint to resume from"}
    )


@dataclass
class SFTScriptArguments(GRPOScriptArguments):
    """
    Script arguments for the SFT training script.
    """
    pass