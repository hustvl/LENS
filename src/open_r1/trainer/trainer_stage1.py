import json
import os
import textwrap
import numpy as np
import cv2
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
import torch
import transformers
import torch.nn as nn
from torch.utils.data import Sampler
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.utils import is_peft_available
from torch.utils.data import Dataset, IterableDataset
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from src.open_r1.arguments import GRPOScriptArguments
if is_peft_available():
    from peft import PeftConfig, get_peft_model
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration, SAMR1ForConditionalGeneration_qwen2p5

from src.segment_anything_2.sam2.build_sam import build_sam2
import torch.nn.functional as F
from src.open_r1.trainer.sam_loss import dice_loss, sigmoid_bce_loss

# optimizer about
from transformers.utils import is_sagemaker_mp_enabled
import logging
logger = logging.getLogger(__name__)


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).
    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def mask2boundary(bin_mask):
    assert bin_mask.max()<=1
    assert bin_mask.min()>=0
    bin_mask = bin_mask.float()
    boudary_width=min(bin_mask.shape[-2:])//20
    if boudary_width%2 ==0:
        boudary_width+=1
    inner_area = 1-F.max_pool2d(1-bin_mask, boudary_width, 1, boudary_width//2)
    boundary_area = (bin_mask - inner_area)>0.5
    return boundary_area

class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
            script_args: GRPOScriptArguments = None, 
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.rec_loss_ratio = script_args.rec_loss_ratio
        self.res_loss_ratio = script_args.res_loss_ratio
        self.if_detach_res_loss = script_args.if_detach_res_loss
        print("self.if_detach_res_loss:", self.if_detach_res_loss)

        self.if_freeze_llm = script_args.if_freeze_llm
        print("self.if_freeze_llm:", self.if_freeze_llm)

        self.if_use_pixel_reward = script_args.if_use_pixel_reward
        print("self.if_use_pixel_reward:", self.if_use_pixel_reward)

        # Models
        # Trained model
        self.script_args = script_args
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )

            if "Qwen2-VL" in model_id:
                model = SAMR1ForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                model.config.num_of_query = model_init_kwargs["num_of_query"]
                model.config.if_use_qwen_connector = model_init_kwargs["if_use_qwen_connector"]
                model.config.if_include_sam = model_init_kwargs["if_include_sam"]
                if self.if_freeze_llm:
                    model.visual.requires_grad_(False)
                    model.model.requires_grad_(False)
                    model.lm_head.requires_grad_(False)
            elif "Qwen2.5-VL" in model_id or "qwen2p5" in model_id:
                model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(model, **model_init_kwargs)
                model.config.num_of_query = model_init_kwargs["num_of_query"]
                model.config.if_use_qwen_connector = model_init_kwargs["if_use_qwen_connector"]
                model.config.if_include_sam = model_init_kwargs["if_include_sam"]
                if self.if_freeze_llm:
                    model.visual.requires_grad_(False)
                    model.model.requires_grad_(False)
                    model.lm_head.requires_grad_(False)
            elif "llava" in model_id:
                model_init_kwargs.pop("use_cache")
                model = LlavaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        # vision_model_params = model.visual.parameters()
        # set_requires_grad(vision_model_params, False)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # meta query
        self.keep_query_grounding = False
        self.if_meta_query = True

        # sam & projector
        self.sam = build_sam2("sam2_hiera_l.yaml", "./pretrained/sam2_hiera_large.pt", device="cuda")
        
        if args.bf16:
            torch_dtype = torch.bfloat16
        elif args.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        self.sam = self.sam.to(dtype=torch_dtype)
        
        input_size = 1024
        self._bb_feat_sizes = [
            (input_size//4, input_size//4),
            (input_size//8, input_size//8),
            (input_size//16, input_size//16),
        ]

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or "llava" in model_id or "qwen2p5" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id


        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper, 8

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        set_seed(args.seed, device_specific=True)
            
        self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation
        self.accelerator.wait_for_everyone()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''
        inputs: [dict_keys(['image', 'problem', 'solution', 'prompt', 'sam_image', 'mask', 'image_path', 'ref_meta']) * batch_size]
        '''
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        device = self.accelerator.device

        prompts_text = []
        # boxes = []
        # points = []
        for x in inputs:
            prompt = json.loads(x['prompt'])
            problem = x["problem"]
            if type(x["solution"]) is str:
                box = eval(x["solution"])
            else:
                box = x["solution"]
            prompt.append(
                {
                    "role": "assistant",
                    "content": f"<|object_ref_start|>{problem}.<|object_ref_end|><|box_start|>({box[0]},{box[1]}),({box[2]},{box[3]})<|box_end|>"
                }
            )
            
            prompts_text.append(
                self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
            )

        # remove the last '\n' in prompts_text
        new_prompt_text = [prompt_text[:-1] for prompt_text in prompts_text if prompt_text[-1] == "\n"]
        prompts_text = new_prompt_text
        images = [x["image"] for x in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)  # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])        
        model_output = model(
            use_learnable_query=self.if_meta_query,
            output_hidden_states=True,
            **prompt_inputs
        )
        hidden_states = model_output.hidden_states

        last_hidden_state = hidden_states[-1]
        if self.if_detach_res_loss:
            last_hidden_state = last_hidden_state.detach()
        box_end_embedding = model.get_sam_embedding(last_hidden_state, if_detach_res_loss=False)

        # SAM forward
        sam_images = torch.stack([x["sam_image"] for x in inputs]).to(box_end_embedding)
        
        backbone_out = self.sam.forward_image(sam_images)
        # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        _, image_embeddings, _, _ = self.sam._prepare_backbone_features(backbone_out)
        image_embeddings = [_.to(sam_images.dtype) for _ in image_embeddings]
        batch_size = sam_images.shape[0]
        if self.sam.directly_add_no_mem_embed:
            image_embeddings[-1] = image_embeddings[-1] + self.sam.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
       
        def save_wi_heatmap(tensor, filename, vmin=0.0, vmax=1.0):
            array = tensor.squeeze().detach().cpu().numpy()
            array = np.clip(array, vmin, vmax)
            array = ((array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(array, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join('./weight_vis', filename), heatmap)

        mask_bce_loss = 0
        mask_dice_loss = 0
        mask_pixel_bce_losses = []
        mask_boundarys = []
        for i in range(len(box_end_embedding)):
            sparse_embeddings, dense_embeddings = self.sam.sam_prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=box_end_embedding[i].unsqueeze(0),
            )
            sparse_embeddings = sparse_embeddings.to(box_end_embedding[i].dtype)
            high_res_features = [
                feat_level[i].unsqueeze(0)
                for feat_level in _features["high_res_feats"]
            ]
            low_res_masks, _, _, _ = self.sam.sam_mask_decoder(
                image_embeddings=_features["image_embed"][i].unsqueeze(0),
                image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image = True,
                high_res_features=high_res_features,
            )
            pred_mask = F.interpolate(low_res_masks.float(), inputs[i]["mask"].shape[-2:], mode="bilinear", align_corners=False)
            gt_mask = inputs[i]["mask"].unsqueeze(0).unsqueeze(0).float()

            loss_multidice = dice_loss(pred_mask, gt_mask, 1., True)
            loss_multimask = sigmoid_bce_loss(pred_mask, gt_mask, 1., loss_on_multimask=True)
            if loss_multimask.size(1) > 1:
                # take the mask indices with the smallest focal + dice loss for back propagation
                loss_combo = self.bce_loss_weight * loss_multimask + self.dice_loss_weight * loss_multidice
                best_loss_inds = torch.argmin(loss_combo, dim=-1)
                batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)

                loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
                loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            else:
                loss_mask = loss_multimask
                loss_dice = loss_multidice
                
            mask_bce_loss += loss_mask.sum()
            mask_dice_loss += loss_dice.sum()

            if self.if_use_pixel_reward:
                pixel_bce_loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
                mask_pixel_bce_losses.append(pixel_bce_loss)
                pred_mask = torch.sigmoid(pred_mask)
                pred_boundary = mask2boundary(pred_mask)
                gt_boundary = mask2boundary(gt_mask)
                mask_boundary = torch.logical_or(pred_boundary, gt_boundary)
                mask_boundarys.append(mask_boundary)
                
        if self.if_use_pixel_reward:
            all_values = torch.cat([t.flatten() for t in mask_pixel_bce_losses])
            batch_bce_mean = all_values.mean()
            batch_bce_std = all_values.std()
            # norm
            mask_pixel_bce_losses = [((pixel_bce_loss - batch_bce_mean) / batch_bce_std) for pixel_bce_loss in mask_pixel_bce_losses]

            all_values = torch.cat([t.flatten() for t in mask_pixel_bce_losses])
            threshold = all_values.mean()
            beta = 0.7
            gamma = 3
            mask_pixel_bce_loss = 0
            for i, mask_pixel_bce in enumerate(mask_pixel_bce_losses):
                num_pixels = mask_pixel_bce.numel()
                # soft discount
                d_i = 1 - beta * torch.sigmoid(gamma * torch.abs(mask_pixel_bce - threshold))
                # d_i = 1 - beta * torch.sigmoid(gamma * torch.relu(mask_pixel_bce - threshold))
                # d_i = 1 - beta * torch.sigmoid(gamma * torch.relu(threshold - mask_pixel_bce))

                high_entropy = mask_pixel_bce > threshold
                mask_boundary = mask_boundarys[i]
                hard_boundary_set = torch.logical_and(mask_boundary, high_entropy)
                
                delta_B = num_pixels - (d_i < 1).sum()
                delta = delta_B / hard_boundary_set.sum()
                w_i = torch.where(hard_boundary_set, d_i + delta, d_i)
                w_i = torch.clip(w_i, 0.1, 3.0)
                weights = w_i * (num_pixels / w_i.sum())

                refid = inputs[i]["ref_meta"]["ref_id"]
                save_wi_heatmap(weights, f'weights_{refid}.png', vmin=0.5, vmax=1.5)

                mask_pixel_bce_loss += (mask_pixel_bce * weights).mean()
                
            mask_bce_loss = mask_pixel_bce_loss / (len(inputs) + 1e-8)            
        else:
            mask_bce_loss = mask_bce_loss / (len(inputs) + 1e-8)
        mask_dice_loss = mask_dice_loss / (len(inputs) + 1e-8)
        res_loss = mask_bce_loss + mask_dice_loss
        self._metrics["res_loss"].append(res_loss.item())
       
        return res_loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def on_train_begin(self):
        found = False
        for i, param_group in enumerate(self.optimizer.param_groups):
            for param in param_group["params"]:
                if hasattr(self.model, "module"):
                    if param is self.model.module.learnable_query:
                        found = True
                        print(f"Found learnable_query in param_group {i} with lr={param_group['lr']}")
                else:
                    if param is self.model.learnable_query:
                        found = True
                        print(f"Found learnable_query in param_group {i} with lr={param_group['lr']}")
        
        if not found:
            print("WARNING: learnable_query not found in any optimizer param_group!") 
