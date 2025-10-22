import json
import os
import deepspeed
import PIL
import torch
import transformers
import torch.distributed as dist
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from torch.utils.data import Sampler
from accelerate.utils import gather, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from torch.utils.data import Dataset, IterableDataset
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed
from trl.trainer.grpo_config import GRPOConfig
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from src.open_r1.trainer.utils import pad
from src.open_r1.arguments import GRPOScriptArguments
from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration, SAMR1ForConditionalGeneration_qwen2p5
from src.segment_anything_2.sam2.build_sam import build_sam2
import torch.nn.functional as F
from src.open_r1.trainer.sam_loss import dice_loss, sigmoid_bce_loss, calculate_boundary_iou

# optimizer about
import logging
logger = logging.getLogger(__name__)

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
local_rank = int(os.environ.get("LOCAL_RANK", -1))

class RepeatRandomSampler(Sampler):
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

class RepeatSampler(Sampler):
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
            for idx in range(self.num_samples)
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count
    

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    target_length = max_length if max_length is not None and max_length > response_length else response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


class Qwen2VLGRPOTrainer(Trainer):
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
            peft_config = None,
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

        # add
        self.if_use_mask_iou_reward = script_args.if_use_mask_iou_reward
        print("self.if_use_mask_iou_reward:", self.if_use_mask_iou_reward)

        self.if_use_boundary_iou_reward = script_args.if_use_boundary_iou_reward
        print("self.if_use_boundary_iou_reward:", self.if_use_boundary_iou_reward)

        self.if_square_mask_iou_as_reward = script_args.if_square_mask_iou_as_reward
        print("self.if_squre_mask_iou_as_reward:", self.if_square_mask_iou_as_reward)

        self.if_square_boundary_iou_as_reward = script_args.if_square_boundary_iou_as_reward
        print("self.if_squre_boundary_iou_as_reward:", self.if_square_boundary_iou_as_reward)

        self.if_use_pixel_reward = script_args.if_use_pixel_reward
        print("self.if_use_pixel_reward:", self.if_use_pixel_reward)

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
            model_init_kwargs["torch_dtype"] = torch.bfloat16

            if "Qwen2-VL" in model_id or ("stage1" in model_id and "2p5" not in model_id):
                model = SAMR1ForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                model.sam = build_sam2("sam2_hiera_l.yaml", "./pretrained/sam2_hiera_large.pt")
                model.sam.requires_grad_(False)
                model.sam.sam_prompt_encoder.requires_grad_(True)
                model.sam.sam_mask_decoder.requires_grad_(True)
                if self.if_freeze_llm:
                    model.visual.requires_grad = False
                    model.model.requires_grad = False
                    model.lm_head.requires_grad = False
                model_init_kwargs = {k:v for k,v in model_init_kwargs.items() if k=="attn_implementation"}              
            elif "Qwen2.5-VL" in model_id or "2p5" in model_id:
                model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(model, **model_init_kwargs)
                print(f'qwen init kwargs = {model_init_kwargs}')
                model.sam = build_sam2("sam2_hiera_l.yaml", "./pretrained/sam2_hiera_large.pt")
                model.sam.requires_grad_(False)
                model.sam.sam_prompt_encoder.requires_grad_(True)
                model.sam.sam_mask_decoder.requires_grad_(True)  
                if self.if_freeze_llm:
                    model.visual.requires_grad = False
                    model.model.requires_grad = False
                    model.lm_head.requires_grad = False
                model_init_kwargs = {k:v for k,v in model_init_kwargs.items() if k=="attn_implementation"}
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

        # set extra training config to model
        model.set_if_detach_res_loss(self.if_detach_res_loss)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id or ("stage1" in model_id and "2p5" not in model_id):
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, 
                    **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id or "2p5" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "llava" in model_id:
                self.ref_model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # meta query
        self.keep_query_grounding = False
        self.if_meta_query = True

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or "llava" in model_id or 'stage1' in model_id or "qwen2p5" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "stage1" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

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

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if global_batch_size % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if "2p5" in model_id:
            model_path = "./pretrained/Qwen/Qwen2.5-VL-3B-Instruct"
        else:
            model_path = "./pretrained/Qwen/Qwen2-VL-2B-Instruct"

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.2,
            distributed_executor_backend="external_launcher",
            enable_prefix_caching=True,
            seed=self.accelerator.process_index,
            max_num_batched_tokens=4096,
            max_model_len=self.max_prompt_length+self.max_completion_length,
        )
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
            max_tokens=self.max_completion_length,
            logprobs=0,            
        )

        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # We need a custom sampler that samples the same prompt multiple times
    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        if self.script_args.if_use_cl==True:
            print("applying curriculum learning")
            return RepeatSampler(self.train_dataset, self.num_generations)
        else:
            return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)
    
    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        for x in inputs:
            x['prompt'] = json.loads(x['prompt'])
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        batch_size = 1
        batched_inputs = {
            k: v.repeat(batch_size, *[1] * (v.dim() - 1)) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }

        if self.max_prompt_length is not None:
            batched_inputs["input_ids"] = batched_inputs["input_ids"][:, -self.max_prompt_length:]
            batched_inputs["attention_mask"] = batched_inputs["attention_mask"][:, -self.max_prompt_length:]
        
        prompt_length = batched_inputs["input_ids"].size(1)
        inputs_vllm = []

        for image_data, messages in zip(images, prompts):
            prompt = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages) if not isinstance(image_data, PIL.Image.Image) else (image_data, None)
            for i in range(batch_size):
                inputs_vllm.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image_data
                    },
                })

        if self.state.global_step != self._last_loaded_step:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                # remove_hooks(model)
                unwrapped_model = self.accelerator.unwrap_model(model)
                if is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    new_state_dict = {}
                    for key in state_dict.keys():
                        if not ("connector" in key or "proj_to_sam" in key or "learnable_query" in key or "conv_1d" in key or "sam" in key):
                            new_state_dict[key] = state_dict[key]
                    
                    llm_model.load_weights(new_state_dict.items())
            
            # Reset cache on vLLM
            self.llm.reset_prefix_cache()
            self._last_loaded_step = self.state.global_step

        all_outputs = self.llm.generate(inputs_vllm, sampling_params=self.sampling_params, use_tqdm=False)       

        # if dist.get_rank() == 0:
        #     import ipdb;ipdb.set_trace()
        # dist.barrier()

        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([batched_inputs["input_ids"], completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id # 151645
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([batched_inputs["attention_mask"], completion_mask], dim=1)

        def get_per_token_logps(model, output_hidden_states=False, **inputs):
            # inputs.keys() contains 'inputs_ids', 'attention_mask', 'pixel_values', and 'image_grid_thw'
            original_input_ids = inputs['input_ids']

            if self.if_meta_query and hasattr(model, 'learnable_query'):
                model_output, low_res_masks = model(
                    use_learnable_query=self.if_meta_query,
                    output_hidden_states=True,
                    **inputs
                )
            else:
                model_output = model(**inputs)

            logits = model_output.logits  # (B, L, V)

            if not self.keep_query_grounding and self.if_meta_query and hasattr(model, 'learnable_query'):
                logits = logits[:, :-self.model.config.num_of_query-1, :] # 额外去除了 learnable query
            else:
                logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
                # NOTE: Still in progress, need to debug for the following code
            
            input_ids = original_input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            per_token_entropys = []
            assert logits.shape[1] == input_ids.shape[1], f"logits shape: {logits.shape}, input_ids shape: {input_ids.shape}"
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                probs = logits_row.softmax(dim=-1)
                entropy = -torch.sum(log_probs*probs, dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
                per_token_entropys.append(entropy)
            if output_hidden_states:
                return torch.stack(per_token_logps), torch.stack(per_token_entropys), low_res_masks
            return torch.stack(per_token_logps), torch.stack(per_token_entropys)

        batched_inputs1 = batched_inputs.copy()
        batched_inputs1["input_ids"] = prompt_completion_ids
        batched_inputs1["attention_mask"] = attention_mask
        batched_inputs1["sam_images"] = torch.stack([x["sam_image"] for x in inputs])
        per_token_logps, per_token_entropys, low_res_masks = get_per_token_logps(model, output_hidden_states=True, **batched_inputs1)        

        batched_inputs1.pop("sam_images")

        # RES loss
        mask_bce_loss = 0
        mask_dice_loss = 0
        segmentation_ious = []
        boundary_ious = []
        mask_bce_losses = []
        
        for i in range(len(low_res_masks)):
            # iou_predictions_list.append(iou_predictions)
            pred_mask = low_res_masks[i]
            pred_mask = F.interpolate(pred_mask.float(), size=inputs[i]["mask"].shape[-2:], mode='bilinear', align_corners=False)

            loss_multidice = dice_loss(pred_mask, inputs[i]["mask"].unsqueeze(0).unsqueeze(0).float(), 1., True)
            if self.if_use_pixel_reward:
                mask_bce_losses.append(F.binary_cross_entropy_with_logits(pred_mask, inputs[i]["mask"].unsqueeze(0).unsqueeze(0).float(), reduction="none"))
            loss_multimask = sigmoid_bce_loss(pred_mask, inputs[i]["mask"].unsqueeze(0).unsqueeze(0).float(), 1., loss_on_multimask=True)
            temp_pred_mask = (pred_mask[0][0]>0)*1.
            temp_gt_mask = inputs[i]["mask"]*1.            
            
            intersection = (temp_pred_mask * temp_gt_mask).sum()
            union = temp_pred_mask.sum() + temp_gt_mask.sum() - intersection
            this_iou = intersection / (union + 1e-6)
            segmentation_ious.append(this_iou.item())

            boundary_iou, _, _ = calculate_boundary_iou(temp_pred_mask.unsqueeze(0).unsqueeze(0), inputs[i]["mask"].unsqueeze(0).unsqueeze(0))
            boundary_ious.append(boundary_iou)

            if loss_multimask.size(1) > 1:
                # take the mask indices with the smallest focal + dice loss for back propagation
                loss_combo = self.bce_loss_weight * loss_multimask + self.dice_loss_weight * loss_multidice
                best_loss_inds = torch.argmin(loss_combo, dim=-1)
                batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)

                loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
                loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
                # loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_mask = loss_multimask
                loss_dice = loss_multidice
                # loss_iou = loss_multiiou
                
            mask_bce_loss += loss_mask.sum()
            mask_dice_loss += loss_dice.sum()
            # iou_pred_loss += loss_iou.sum()

        if self.if_use_pixel_reward:
            mask_bce_losses = torch.cat(mask_bce_losses)
            with torch.no_grad():
                slices = torch.logical_and(mask_bce_losses < 0.7, mask_bce_losses > 0.0)

            mask_bce_loss = mask_bce_losses[slices].mean()
        else:
            mask_bce_loss = mask_bce_loss / (len(inputs) + 1e-8)
            
        mask_dice_loss = mask_dice_loss / (len(inputs) + 1e-8)
        res_loss = mask_bce_loss + mask_dice_loss
        self._metrics["res_loss"].append(res_loss.item())
       
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        per_token_entropys = per_token_entropys[:, prompt_length - 1:]
        ppl = -((per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() 

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps, ref_per_token_entropys = get_per_token_logps(self.ref_model, **batched_inputs1)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps, ref_per_token_entropys = get_per_token_logps(model, **batched_inputs1)

        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        ref_per_token_entropys = ref_per_token_entropys[:, prompt_length - 1:]
        ref_ppl = -((ref_per_token_logps * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Compute the KL divergence between the model and the reference model
        k1 = per_token_logps - ref_per_token_logps
        k3 = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        kimi = 0.5*(per_token_logps - ref_per_token_logps)**2
        
        # select the kl approximator
        if self.script_args.kl_approximator == 'k3':
            per_token_kl = k3
        elif self.script_args.kl_approximator == 'k1':
            per_token_kl = k1
        elif self.script_args.kl_approximator in ['kimikl', 'fullkimi']:
            per_token_kl = kimi

        # Compute the entropy reg loss
        entropy_loss = -per_token_entropys
        ref_entropy_loss = -ref_per_token_entropys

        # Decode the generated completions, skip_special_tokens=False when rec task
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=self.script_args.skip_special_tokens)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                reward_kwargs['current_step'] = self.state.global_step
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if self.if_use_mask_iou_reward:
            new_rewards_per_func = torch.zeros(len(prompts), rewards_per_func.size(1) + 1, device=device, dtype=rewards_per_func.dtype)

            new_rewards_per_func[:, -1] = torch.tensor(segmentation_ious, dtype=torch.float32, device=device)
            if self.if_square_mask_iou_as_reward:
                new_rewards_per_func[:, -1] = new_rewards_per_func[:, -1]**2

            new_rewards_per_func[:, :-1] = rewards_per_func

            rewards_per_func = new_rewards_per_func

        if self.if_use_boundary_iou_reward:
            new_rewards_per_func = torch.zeros(len(prompts), rewards_per_func.size(1) + 1, device=device, dtype=rewards_per_func.dtype)

            new_rewards_per_func[:, -1] = torch.tensor(boundary_ious, dtype=torch.float32, device=device)
            if self.if_square_boundary_iou_as_reward:
                new_rewards_per_func[:, -1] = new_rewards_per_func[:, -1]**2

            new_rewards_per_func[:, :-1] = rewards_per_func

            rewards_per_func = new_rewards_per_func   

        rewards_per_func = gather(rewards_per_func)
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        rewards = self.script_args.reward_scale*rewards

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        
        # compute advantages
        if self.script_args.no_mean_for_same_reward:
            std_mask  = torch.tensor([1.0 if std.item() != 0 else 0.0 for std in std_grouped_rewards], device=self.accelerator.device)
            advantages = (rewards - mean_grouped_rewards*std_mask) / (std_grouped_rewards + 1e-4)
            advantages = torch.clamp(advantages, min=-1.0, max=1.0)
        else:
            if self.script_args.kl_approximator == 'fullkimi':
                advantages = rewards - mean_grouped_rewards
            else:   
                advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        
        # x - x.detach() allows for preserving gradients from x
        if self.script_args.kl_approximator == 'fullkimi':
            per_token_loss = -torch.exp(per_token_logps) * advantages.unsqueeze(1)
        else:
            per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

        if self.script_args.use_kl:
            per_token_loss += self.beta * per_token_kl

        if self.script_args.entropy_reg:
            per_token_loss += self.script_args.entropy_weight * entropy_loss
        
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log the metrics
        completion_length = completion_mask.sum(1).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        if self.if_use_mask_iou_reward:
            self._metrics[f"rewards/mask_iou_reward"].append(reward_per_func[-1].item())

        # log reward metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        # log kl metrics
        mean_k1_kl = ((k1 * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_k3_kl = ((k3 * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kimi_kl = ((kimi * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["k1_kl"].append(self.accelerator.gather_for_metrics(mean_k1_kl).mean().item())
        self._metrics["k3_kl"].append(self.accelerator.gather_for_metrics(mean_k3_kl).mean().item())
        self._metrics["kimi_kl"].append(self.accelerator.gather_for_metrics(mean_kimi_kl).mean().item())
        
        # log entropy metrics
        mean_entropy_loss = ((entropy_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()   
        mean_ref_entropy_loss = ((ref_entropy_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics['entropy_loss'].append(self.accelerator.gather_for_metrics(mean_entropy_loss).mean().item())
        self._metrics['delta_ref_entropy_loss'].append(self.accelerator.gather_for_metrics(mean_entropy_loss-mean_ref_entropy_loss).mean().item())
        
        # log ppl and other metrics
        self._metrics['ppl'].append(self.accelerator.gather_for_metrics(ppl).mean().item())
        self._metrics['delta_ref_ppl'].append(self.accelerator.gather_for_metrics(ppl-ref_ppl).mean().item())
        self._metrics['advantages'].append(self.accelerator.gather_for_metrics(advantages.mean()).mean().item())
        if self.accelerator.is_main_process:
            self._metrics["temperature"].append(self.sampling_params.temperature)

        return loss * self.rec_loss_ratio + res_loss * self.res_loss_ratio

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
