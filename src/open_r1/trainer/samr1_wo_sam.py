# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.utils import logging
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLCausalLMOutputWithPast,
)
from src.open_r1.trainer.qwen2_vl import Qwen2VLVisionConnectorSimple

logger = logging.get_logger(__name__)


class SAMR1Config(Qwen2VLConfig):
    def __init__(self, num_of_query=None, if_use_qwen_connector=None, **kwargs):
        super().__init__(**kwargs)
        self.num_of_query = num_of_query if num_of_query else None
        self.if_use_qwen_connector = if_use_qwen_connector if if_use_qwen_connector else None


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
        self.learnable_query = nn.Parameter(torch.randn(1, model_num_of_query, config.hidden_size))
        self.learnable_query.ds_full_param = True  # For DeepSpeed full parameter storage

        self.model_num_of_query = model_num_of_query
        self.model_if_use_qwen_connector = model_if_use_qwen_connector

        self.conv_1d = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=model_num_of_query)
        
        if model_if_use_qwen_connector:
            self.connector = Qwen2VLVisionConnectorSimple(depth=4, seq_len=model_num_of_query, embed_dim=config.hidden_size)

        self.proj_to_sam = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 256)
        )   

        self._init_custom_params()
        self.post_init()
    
    def _init_custom_params(self):
        """Initialize custom parameters."""
        nn.init.normal_(self.learnable_query, mean=0.0, std=0.02)
        nn.init.normal_(self.conv_1d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.conv_1d.bias)
    
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
        """
        Extends Qwen2VL forward method to support learnable queries.
        """
        if use_learnable_query:
            attention_mask, inputs_embeds = self.process_llm_input(
                input_ids, pixel_values, image_grid_thw, attention_mask
            )
            input_ids = None

        return super().forward(
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
    
    def process_llm_input(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        """
        Map input_ids to embeddings, replace image tokens with vision features, 
        and append learnable query tokens at the end.
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
            attention_mask = torch.cat([
                attention_mask.to(inputs_embeds.device),
                torch.ones(attention_mask.size(0), self.model_num_of_query).to(inputs_embeds.device)
            ], 1)
        else:
            attention_mask = torch.ones(inputs_embeds.size(0), inputs_embeds.size(1)).to(inputs_embeds.device)
        
        return attention_mask, inputs_embeds

    def get_sam_embedding(self, hidden_states, if_detach_res_loss=False):
        """
        Extract SAM embedding from hidden states.
        """
        query_hidden_state = hidden_states[:, -self.model_num_of_query:]

        if self.model_if_use_qwen_connector:
            query_hidden_state = self.connector(query_hidden_state)

        query_hidden_state = self.conv_1d(query_hidden_state.transpose(1, 2)).transpose(1, 2).contiguous()
        sam_embedding = self.proj_to_sam(query_hidden_state)

        if if_detach_res_loss:
            query_hidden_state = query_hidden_state.detach()
         
        return sam_embedding
    
    def postprocess_masks(self, masks, orig_hw):
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


__all__ = ["SAMR1ForConditionalGeneration"]
