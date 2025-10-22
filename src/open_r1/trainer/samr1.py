# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SAM-R1 model based on Qwen2-VL."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLCausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLCausalLMOutputWithPast,
)
from src.open_r1.trainer.qwen2_vl import Qwen2VLVisionConnectorSimple
from src.segment_anything_2.sam2.build_sam import build_sam2

logger = logging.get_logger(__name__)
local_rank = int(os.getenv("LOCAL_RANK", -1))


class SAMR1Config(Qwen2VLConfig):
    def __init__(self, num_of_query=None, if_use_qwen_connector=None, if_include_sam=None, **kwargs):
        super().__init__(**kwargs)
        self.num_of_query = num_of_query
        self.if_use_qwen_connector = if_use_qwen_connector


class SAMR1ForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """
    SAM-R1 model for conditional generation based on Qwen2VL.
    This model integrates a learnable query parameter and projection to SAM.
    """
    config_class = SAMR1Config
    
    def __init__(self, config, num_of_query=1, if_use_qwen_connector=False, **kwargs):
        super().__init__(config)
        model_num_of_query = config.num_of_query or num_of_query
        model_if_use_qwen_connector = config.if_use_qwen_connector or if_use_qwen_connector

        self.if_detach_res_loss = False
        print("Num of Query:", model_num_of_query)

        self.learnable_query = nn.Parameter(torch.randn(1, model_num_of_query, config.hidden_size), requires_grad=True)
        # Set ds_full_param to True so DeepSpeed will not shard this parameter
        self.learnable_query.ds_full_param = True
        self.learnable_query.ds_persist = True

        self.model_num_of_query = model_num_of_query
        self.model_if_use_qwen_connector = model_if_use_qwen_connector

        self.conv_1d = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=model_num_of_query)
        
        if model_if_use_qwen_connector:
            self.connector = Qwen2VLVisionConnectorSimple(depth=4, seq_len=model_num_of_query, embed_dim=config.hidden_size)

        # Projection layer to SAM
        self.proj_to_sam = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 256)
        )   

        self.sam = build_sam2("sam2_hiera_l.yaml", device=self.model.device)
        del self.sam.maskmem_tpos_enc
        del self.sam.memory_attention
        del self.sam.memory_encoder

        input_size = 1024
        self._bb_feat_sizes = [
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16),
        ]
        
        self._init_custom_params()
        self.post_init()
    
    def _init_custom_params(self):
        """Initialize custom parameters"""
        nn.init.normal_(self.learnable_query, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_1d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv_1d.bias)

    def set_if_detach_res_loss(self, if_detach_res_loss):
        self.if_detach_res_loss = if_detach_res_loss
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_learnable_query: bool = False,
        **kwargs,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        # Process learnable query
        if use_learnable_query:
            attention_mask, inputs_embeds = self.process_llm_input(input_ids, pixel_values, image_grid_thw, attention_mask)
            input_ids = None

        sam_images = kwargs.pop("sam_images", None)

        # Call the parent forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if sam_images is not None:
            assert output_hidden_states is True
            box_end_embedding = self.get_sam_embedding(outputs.hidden_states[-1], if_detach_res_loss=self.if_detach_res_loss)
            sam_images = sam_images.to(box_end_embedding)
            backbone_out = self.sam.forward_image(sam_images)
            _, image_embeddings, _, _ = self.sam._prepare_backbone_features(backbone_out)
            image_embeddings = [_.to(sam_images.dtype) for _ in image_embeddings]
            batch_size = sam_images.shape[0]
            if self.sam.directly_add_no_mem_embed:
                image_embeddings[-1] += self.sam.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
            ][::-1]
            _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

            pred_masks = []
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
                    repeat_image=True,
                    high_res_features=high_res_features,
                )
                pred_masks.append(low_res_masks)
            return outputs, pred_masks

        return outputs
    
    def process_llm_input(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        """
        To append metaquery, the LLM needs to use inputs_embeds instead of input_ids.
        This manually maps input_ids to inputs_embeds and appends metaquery at the end.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        inputs_embeds = torch.cat([
            inputs_embeds, 
            self.learnable_query.repeat(inputs_embeds.size(0), 1, 1)
        ], 1)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(attention_mask.size(0), self.model_num_of_query).to(attention_mask)
            ], 1)
        else:
            attention_mask = torch.ones(inputs_embeds.size(0), inputs_embeds.size(1)).to(inputs_embeds.device)
        
        return attention_mask, inputs_embeds

    def get_sam_embedding(self, hidden_states, if_detach_res_loss=False):
        """
        Extract SAM embedding from hidden states.
        
        Args:
            hidden_states: Model hidden states.
        Returns:
            Embedding projected into SAM space.
        """
        query_hidden_state = hidden_states[:, -self.model_num_of_query:]

        if if_detach_res_loss:
            query_hidden_state = query_hidden_state.detach()

        if self.model_if_use_qwen_connector:
            query_hidden_state = self.connector(query_hidden_state)

        query_hidden_state = self.conv_1d(query_hidden_state.transpose(1, 2)).transpose(1, 2).contiguous()
        sam_embedding = self.proj_to_sam(query_hidden_state)
        return sam_embedding
    
    def postprocess_masks(self, masks, orig_hw):
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


class SAMR1ForConditionalGeneration_qwen2p5(Qwen2_5_VLForConditionalGeneration):
    """
    SAM-R1 model for conditional generation based on Qwen2VL.
    Integrates a learnable query parameter and projection to SAM for joint vision-language tasks.
    """
    config_class = SAMR1Config
    
    def __init__(self, config, num_of_query=64, if_use_qwen_connector=True, **kwargs):
        super().__init__(config)
        model_num_of_query = config.num_of_query or num_of_query
        model_if_use_qwen_connector = config.if_use_qwen_connector or if_use_qwen_connector

        self.if_detach_res_loss = False

        # Learnable context queries
        self.learnable_query = nn.Parameter(torch.randn(1, model_num_of_query, config.hidden_size), requires_grad=True)
        self.learnable_query.ds_full_param = True  # Keep full param in DeepSpeed ZeRO
        self.learnable_query.ds_persist = True

        self.model_num_of_query = model_num_of_query
        self.model_if_use_qwen_connector = model_if_use_qwen_connector

        self.conv_1d = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=model_num_of_query)
        
        if model_if_use_qwen_connector:
            self.connector = Qwen2VLVisionConnectorSimple(depth=4, seq_len=model_num_of_query, embed_dim=config.hidden_size)

        # Projection to SAM feature space
        self.proj_to_sam = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 256)
        )   

        # Build SAM backbone
        self.sam = build_sam2("sam2_hiera_l.yaml", device=self.model.device)
        del self.sam.maskmem_tpos_enc
        del self.sam.memory_attention
        del self.sam.memory_encoder

        input_size = 1024
        self._bb_feat_sizes = [
            (input_size // 4, input_size // 4),
            (input_size // 8, input_size // 8),
            (input_size // 16, input_size // 16),
        ]
        
        self._init_custom_params()
        self.post_init()
    
    def _init_custom_params(self):
        """Initialize custom parameters."""
        nn.init.normal_(self.learnable_query, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_1d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv_1d.bias)

    def set_if_detach_res_loss(self, if_detach_res_loss):
        self.if_detach_res_loss = if_detach_res_loss
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_learnable_query: bool = False,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        """
        Extended forward method to support learnable query injection.
        """
        if use_learnable_query:
            attention_mask, inputs_embeds = self.process_llm_input(input_ids, pixel_values, image_grid_thw, attention_mask)
            input_ids = None

        sam_images = kwargs.pop("sam_images", None)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if sam_images is not None:
            assert output_hidden_states is True
            box_end_embedding = self.get_sam_embedding(outputs.hidden_states[-1], if_detach_res_loss=self.if_detach_res_loss)
            sam_images = sam_images.to(box_end_embedding)
            backbone_out = self.sam.forward_image(sam_images)
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

            pred_masks = []
            for i in range(len(box_end_embedding)):
                sparse_embeddings, dense_embeddings = self.sam.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=box_end_embedding[i].unsqueeze(0),
                )
                sparse_embeddings = sparse_embeddings.to(box_end_embedding[i].dtype)
                high_res_features = [feat_level[i].unsqueeze(0) for feat_level in _features["high_res_feats"]]
                low_res_masks, _, _, _ = self.sam.sam_mask_decoder(
                    image_embeddings=_features["image_embed"][i].unsqueeze(0),
                    image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=True,
                    high_res_features=high_res_features,
                )
                pred_masks.append(low_res_masks)
            return outputs, pred_masks
                 
        return outputs
    
    def process_llm_input(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        """
        Convert input_ids to embeddings and append learnable queries at the end.
        """
        if not isinstance(input_ids, torch.LongTensor):
            input_ids = input_ids.to(torch.long)
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        inputs_embeds = torch.cat(
            [inputs_embeds, self.learnable_query.repeat(inputs_embeds.size(0), 1, 1)], dim=1
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(attention_mask.size(0), self.model_num_of_query).to(attention_mask)], dim=1
            )
        else:
            attention_mask = torch.ones(inputs_embeds.size(0), inputs_embeds.size(1)).to(inputs_embeds.device)
        
        return attention_mask, inputs_embeds

    def get_sam_embedding(self, hidden_states, if_detach_res_loss=False):
        """
        Extract and project SAM embedding from the last learnable queries in hidden states.
        """
        query_hidden_state = hidden_states[:, -self.model_num_of_query:]

        if if_detach_res_loss:
            query_hidden_state = query_hidden_state.detach()

        if self.model_if_use_qwen_connector:
            query_hidden_state = self.connector(query_hidden_state)

        query_hidden_state = self.conv_1d(query_hidden_state.transpose(1, 2)).transpose(1, 2).contiguous()
        sam_embedding = self.proj_to_sam(query_hidden_state)
         
        return sam_embedding
    
    def postprocess_masks(self, masks, orig_hw):
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


__all__ = ["SAMR1ForConditionalGeneration", "SAMR1ForConditionalGeneration_qwen2p5"]
