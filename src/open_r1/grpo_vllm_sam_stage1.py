import os

from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
from src.open_r1.constants import reward_funcs_registry
from src.open_r1.utils import save_args_to_txt
from src.open_r1.trainer.trainer_stage1 import Qwen2VLGRPOTrainer
from src.open_r1.arguments import GRPOScriptArguments
from src.open_r1.dataset import ReferSegDataset

# fix bugs in qwen2.5vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple, Optional
from transformers import TrainerCallback


local_rank = int(os.environ.get("LOCAL_RANK", -1))


class GradientCheckerCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        # This callback runs after backward() and before optimizer.step()
        if args.local_rank == 0 or args.local_rank == -1:  # Check if this is the main process
            print(f"\n--- Gradient Check (Callback) Global Step: {state.global_step} ---")
            current_model = model.module if hasattr(model, "module") else model
            params_to_check = []
            if hasattr(current_model, "learnable_query"):
                params_to_check.append(("learnable_query", current_model.learnable_query))
            # Similarly, add proj_to_sam or other parameters here

            for name, param in params_to_check:
                if param.grad is not None:
                    print(f"  Parameter: {name}, Gradient Norm: {param.grad.norm().item():.4e}")
                    if param.grad.norm().item() == 0.0:
                        print(f"  Warning: Gradient norm of {name} is zero.")
                else:
                    print(f"  Parameter: {name}, Gradient is None.")
            print("--- End of Gradient Check (Callback) ---")


def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        if position_embeddings is None:
            print(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

def main(script_args, training_args, model_args):
    # Get reward functions
    if any( '+' in reward for reward in script_args.reward_funcs):
        script_args.reward_funcs = script_args.reward_funcs[0].split('+')
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    
    # prepare parameters for Kimi-KL
    if script_args.kl_approximator == 'fullkimi':
        script_args.use_kl = True
        training_args.sync_ref_model = True
        training_args.ref_model_mixup_alpha = 1.0
        training_args.ref_model_sync_steps = 1
    
    # save args to output_dir
    save_args_to_txt(script_args, os.path.join(training_args.output_dir, 'config', 'script_args.txt'))
    save_args_to_txt(training_args, os.path.join(training_args.output_dir, 'config', 'training_args.txt'))
    save_args_to_txt(model_args, os.path.join(training_args.output_dir, 'config', 'model_args.txt'))

    # Load the dataset
    dataset = ReferSegDataset(script_args)
    
    trainer_cls = Qwen2VLGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        script_args=script_args,
        # callbacks=[SaveSAMCallback(trainer=None)]
    )

    # Train and push the model to the Hub
    # trainer.train(resume_from_checkpoint=script_args.resume_checkpoint)
    trainer.train()

    # trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if 'num_of_query' in script_args.__dict__:
        training_args.model_init_kwargs = {'num_of_query': script_args.num_of_query}

    if 'if_use_qwen_connector' in script_args.__dict__:
        if training_args.model_init_kwargs is None:
            training_args.model_init_kwargs = {'if_use_qwen_connector': script_args.if_use_qwen_connector}
        else:
            training_args.model_init_kwargs['if_use_qwen_connector'] = script_args.if_use_qwen_connector
    
    if 'if_include_sam' in script_args.__dict__:
        if training_args.model_init_kwargs is None:
            training_args.model_init_kwargs = {'if_include_sam': script_args.if_include_sam}
        else:
            training_args.model_init_kwargs['if_include_sam'] = script_args.if_include_sam

    print('training_args:\n', training_args)
    print('script_args:\n', script_args)
    print('model_args:\n', model_args)
    main(script_args, training_args, model_args)
