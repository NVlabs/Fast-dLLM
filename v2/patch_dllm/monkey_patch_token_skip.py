from typing import Optional, Union

import torch

from transformers.generation.utils import GenerateDecoderOnlyOutput  
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple
import sys

from .token_skip_policy import TokenSkipPolicy

apply_rotary_pos_emb = None
CausalLMOutputWithPastAndBlockCache = None
BaseModelOutputWithPastAndBlockCache = None
fused_flex_attention = None
_originals = {}

def patch_token_skip(model):
    global apply_rotary_pos_emb, CausalLMOutputWithPastAndBlockCache, BaseModelOutputWithPastAndBlockCache, fused_flex_attention
    mod = sys.modules[model.__class__.__module__]

    _originals['mod'] = mod
    _originals['attention_forward'] = mod.Fast_dLLM_QwenAttention.forward
    _originals['decoder_forward'] = mod.Fast_dLLM_QwenDecoderLayer.forward
    _originals['model_forward'] = mod.Fast_dLLM_QwenModel.forward
    _originals['causal_forward'] = mod.Fast_dLLM_QwenForCausalLM.forward
    _originals['generate'] = mod.Fast_dLLM_QwenForCausalLM.generate

    apply_rotary_pos_emb = mod.apply_rotary_pos_emb
    CausalLMOutputWithPastAndBlockCache = mod.CausalLMOutputWithPastAndBlockCache
    BaseModelOutputWithPastAndBlockCache = mod.BaseModelOutputWithPastAndBlockCache
    fused_flex_attention = mod.fused_flex_attention
    mod.Fast_dLLM_QwenAttention.forward = Fast_dLLM_QwenAttention_forward
    mod.Fast_dLLM_QwenDecoderLayer.forward = Fast_dLLM_QwenDecoderLayer_forward
    mod.Fast_dLLM_QwenModel.forward = Fast_dLLM_QwenModel_forward
    mod.Fast_dLLM_QwenForCausalLM.forward = Fast_dLLM_QwenForCausalLM_forward
    mod.Fast_dLLM_QwenForCausalLM.generate = Fast_dLLM_QwenForCausalLM_generate

def unpatch_token_skip():
    if not _originals:
        return
    mod = _originals['mod']
    mod.Fast_dLLM_QwenAttention.forward = _originals['attention_forward']
    mod.Fast_dLLM_QwenDecoderLayer.forward = _originals['decoder_forward']
    mod.Fast_dLLM_QwenModel.forward = _originals['model_forward']
    mod.Fast_dLLM_QwenForCausalLM.forward = _originals['causal_forward']
    mod.Fast_dLLM_QwenForCausalLM.generate = _originals['generate']
    _originals.clear()

def Fast_dLLM_QwenAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    update_past_key_values: Optional[bool] = False,
    block_past_key_values: Optional[Cache] = None,
    replace_position: Optional[int] = None,
    skip_policy: Optional[TokenSkipPolicy] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if self.training:
        #split q into two parts
        q_1 = query_states[:,:,:query_states.shape[2]//2]
        q_2 = query_states[:,:,query_states.shape[2]//2:]
        #split k into two parts
        k_1 = key_states[:,:,:key_states.shape[2]//2]
        k_2 = key_states[:,:,key_states.shape[2]//2:]
        q_1, k_1 = apply_rotary_pos_emb(q_1, k_1, cos, sin)
        q_2, k_2 = apply_rotary_pos_emb(q_2, k_2, cos, sin)
        query_states = torch.cat((q_1, q_2), dim=-2)
        key_states = torch.cat((k_1, k_2), dim=-2)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if block_past_key_values is not None:
        if len(block_past_key_values) <= self.layer_idx:
            if skip_policy:
                raise ValueError("block_past_key_values must be filled before skip policy can be used")
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = block_past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            block_cache_key_states = block_past_key_values[self.layer_idx][0]
            block_cache_value_states = block_past_key_values[self.layer_idx][1]

            if skip_policy:
                skip_policy.update_block_cache(block_cache_key_states, key_states, replace_position)
                skip_policy.update_block_cache(block_cache_value_states, value_states, replace_position)
            else:
                block_cache_key_states[:, :, replace_position:replace_position+key_states.shape[2]] = key_states
                block_cache_value_states[:, :, replace_position:replace_position+value_states.shape[2]] = value_states

            key_states = block_cache_key_states
            value_states = block_cache_value_states

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        if update_past_key_values:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        elif len(past_key_value) > self.layer_idx:
            key_states = torch.cat((past_key_value[self.layer_idx][0], key_states), dim=-2)
            value_states = torch.cat((past_key_value[self.layer_idx][1], value_states), dim=-2)

    if self.training:
        attn_output = fused_flex_attention(query_states, key_states, value_states, mask=attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            is_causal=False,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output

def Fast_dLLM_QwenDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    update_past_key_values: Optional[bool] = False,
    use_block_cache: Optional[bool] = False,
    block_past_key_values: Optional[Cache] = None,
    replace_position: Optional[int] = None,
    skip_policy: Optional[TokenSkipPolicy] = None,
    **kwargs
) -> tuple[torch.Tensor]:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        update_past_key_values=update_past_key_values,
        use_block_cache=use_block_cache,
        block_past_key_values=block_past_key_values,
        replace_position=replace_position,
        skip_policy=skip_policy,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

def Fast_dLLM_QwenModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    update_past_key_values: Optional[bool] = False,
    block_size: Optional[int] = 32,
    use_block_cache: Optional[bool] = False,
    block_past_key_values: Optional[Cache] = None,
    replace_position: Optional[int] = None,
    skip_policy: Optional[TokenSkipPolicy] = None,
    compute_counter=None,
    **kwargs
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if use_block_cache and block_past_key_values is None:
        block_past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if self.training:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]//2, device=inputs_embeds.device
            )
        else:
            if use_block_cache:
                block_start_position = past_seen_tokens+replace_position if replace_position is not None else past_seen_tokens
                cache_position = torch.arange(
                    block_start_position, block_start_position + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            else:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1] if not self.training else inputs_embeds.shape[1]//2, device=inputs_embeds.device
                )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if skip_policy:
        total_before = inputs_embeds.shape[0] * inputs_embeds.shape[1]
        hidden_states = skip_policy.filter_embeds(inputs_embeds)
        position_ids = skip_policy.filter_positions(position_ids)
        skipped = total_before - hidden_states.shape[0] * hidden_states.shape[1]
        if compute_counter and skipped > 0:
            compute_counter.log_token_skip(skipped * self.config.num_hidden_layers)
    else:
        hidden_states = inputs_embeds

    if self.training:
        attention_mask = self.gen_mask(labels.shape[1], self.bd_size, labels.shape[0], self.config.num_attention_heads).to(device=inputs_embeds.device)
    else:
        if use_block_cache and block_past_key_values.get_seq_length() != 0:
            attention_mask = None
        else:
            attention_mask = self.eval_mask(input_ids.shape[1], block_size, past_key_values.get_seq_length() if past_key_values is not None else 0).to(device=inputs_embeds.device)

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if compute_counter:
            compute_counter.log_tokens(hidden_states.shape[0] * hidden_states.shape[1])
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            update_past_key_values=update_past_key_values,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            skip_policy=skip_policy,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPastAndBlockCache(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        block_past_key_values=block_past_key_values if use_block_cache else None,
    )

@can_return_tuple
def Fast_dLLM_QwenForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    update_past_key_values: Optional[bool] = False,
    block_size: Optional[int] = 32,
    use_block_cache: Optional[bool] = False,
    block_past_key_values: Optional[Cache] = None,
    replace_position: Optional[int] = None,
    mask_id: Optional[int] = 151665,
    skip_policy: Optional[TokenSkipPolicy] = None,
    compute_counter=None,
    **kwargs
) -> CausalLMOutputWithPastAndBlockCache:

    if self.training:
        original_labels = labels.clone()
        original_input_ids = input_ids.clone()

        noisy_input_ids = input_ids.clone()

        input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
        b, l = input_ids.shape
        t = torch.rand((b,), device=input_ids.device)
        eps=1e-3
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        x_t = torch.where(mask_indices, mask_id, input_ids).reshape(labels.shape)
        noisy_input_ids[labels != -100] = x_t[labels != -100]
        mask = (noisy_input_ids != mask_id)
        labels[mask] = -100
        input_ids = torch.cat([noisy_input_ids, input_ids.reshape(labels.shape)], dim=1)

        complementary_noisy_input_ids = original_input_ids.clone()
        complementary_labels = original_labels.clone()

        complementary_input_ids = original_input_ids.reshape(original_input_ids.shape[0] * original_input_ids.shape[1] // self.model.bd_size, self.model.bd_size)

        complementary_mask_indices = ~mask_indices
        complementary_x_t = torch.where(complementary_mask_indices, mask_id, complementary_input_ids).reshape(labels.shape)
        complementary_noisy_input_ids[complementary_labels != -100] = complementary_x_t[complementary_labels != -100]
        complementary_mask = (complementary_noisy_input_ids != mask_id)
        complementary_labels[complementary_mask] = -100
        complementary_input_ids = torch.cat([complementary_noisy_input_ids, complementary_input_ids.reshape(complementary_labels.shape)], dim=1)

        input_ids = torch.cat([input_ids, complementary_input_ids], dim=0)
        labels = torch.cat([labels, complementary_labels], dim=0)

    if skip_policy is not None and skip_policy.skip_mask.all():
        hidden_states = skip_policy.past_hidden_states[1].clone()
        outputs = BaseModelOutputWithPastAndBlockCache(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            block_past_key_values=block_past_key_values if use_block_cache else None,
        )
        skip_policy.reset_state()
    else:
        outputs: BaseModelOutputWithPastAndBlockCache = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            update_past_key_values=update_past_key_values,
            block_size=block_size,
            use_block_cache=use_block_cache,
            block_past_key_values=block_past_key_values,
            replace_position=replace_position,
            skip_policy=skip_policy,
            compute_counter=compute_counter,
            **kwargs,
        )
        hidden_states = skip_policy.reconstruct_hidden_states(outputs.last_hidden_state) if skip_policy else outputs.last_hidden_state

    if self.training:
        hidden_states = hidden_states[:, :hidden_states.shape[1]//2, :]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPastAndBlockCache(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=hidden_states,
        attentions=outputs.attentions,
        block_past_key_values=outputs.block_past_key_values,
    )

@torch.no_grad()
def Fast_dLLM_QwenForCausalLM_generate(
    self,
    input_ids,
    max_new_tokens=None,
    max_length=None,
    tokenizer=None,
    mask_id=151665,
    threshold=1,
    small_block_size=8,
    block_size=32,
    stop_token=151645,
    stopping_criteria=None,
    top_p=0.95,
    temperature=0,
    use_block_cache=False,
    return_dict_in_generate=False,
    output_scores=False,
    output_hidden_states=False,
    skip_policy=None,
    compute_counter=None,
    **kwargs
):
    if max_new_tokens is None and max_length is None:
        raise ValueError("Either max_new_tokens or max_length must be specified")
    if max_new_tokens is None:
        max_new_tokens = max_length - input_ids.shape[1]
    
    scores_list = [] if output_scores else None
    decoder_hidden_states = [] if output_hidden_states else None
    
    num_blocks = max_new_tokens // block_size
    original_input_length = input_ids.shape[1]

    if input_ids.shape[1] > block_size:
        output = self.forward(
            input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], 
            use_cache=True, 
            update_past_key_values=True, 
            block_size=block_size,
            compute_counter=compute_counter,
        )
        logits, past_key_values = output.logits, output.past_key_values
        
        if output_scores:
            scores_list.append(logits)
        if output_hidden_states and hasattr(output, 'hidden_states'):
            decoder_hidden_states.append(output.hidden_states)
        
        if input_ids.shape[1] % block_size == 0:
            next_token = logits[:, -1:, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    else:
        past_key_values = None

    num_small_blocks = block_size // small_block_size

    for block_idx in range(num_blocks):
        if stop_token in input_ids[:, original_input_length:]:
            break
        prompt_length = input_ids.shape[1]
        # Initialize x_init with mask_id
        x_init = mask_id * torch.ones(
            (input_ids.shape[0], block_size-prompt_length%block_size), 
            device=self.device, 
            dtype=torch.long
        )
        x_init = torch.cat([input_ids, x_init], dim=1)

        x_t = x_init.clone()
        block_past_key_values = None
        
        while True:
            if stop_token in x_t[:, prompt_length:]:
                stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                    break
            mask_idx = (x_t[:, -block_size:] == mask_id)
            
            # Decode a complete block, update cache, and generate the next token
            if mask_idx.sum() == 0:
                output = self.forward(
                    input_ids=x_t[:, -block_size:], 
                    use_cache=True, 
                    past_key_values=past_key_values, 
                    update_past_key_values=True, 
                    block_size=block_size,
                    compute_counter=compute_counter,
                )
                logits, past_key_values = output.logits, output.past_key_values
                
                if output_scores:
                    scores_list.append(logits)
                if output_hidden_states and hasattr(output, 'hidden_states'):
                    decoder_hidden_states.append(output.hidden_states)
                
                next_token = logits[:, -1:, :].argmax(dim=-1)
                x_t = torch.cat([x_t, next_token], dim=1)
                break
                
            for small_block_idx in range(num_small_blocks):
                if skip_policy:
                    skip_policy.reset_state()
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size

                start = -block_size + small_block_start_idx
                end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                
                while True:
                    mask_idx = (x_t[:, -block_size:] == mask_id)
                    if mask_idx[:, start:end].sum() == 0:
                        break
                    if stop_token in x_t[:, prompt_length:]:
                        stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                        if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                            break

                    if compute_counter:
                        compute_counter.log_denoising_iter()

                    active = False
                    if use_block_cache:
                        if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                            if compute_counter:
                                compute_counter.log_full_block()
                            output = self.forward(
                                input_ids=x_t[:, -block_size:], 
                                use_cache=True, 
                                past_key_values=past_key_values, 
                                update_past_key_values=False, 
                                use_block_cache=True,
                                compute_counter=compute_counter,
                            )
                            logits, block_past_key_values = output.logits, output.block_past_key_values
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                            if skip_policy:
                                skip_policy.update_hidden_states(output.hidden_states[:, start:end])
                        else:
                            if skip_policy:
                                active = skip_policy.check_token_skip()
                            output = self.forward(
                                input_ids=x_t[:,start:end], 
                                use_cache=True, 
                                past_key_values=past_key_values, 
                                update_past_key_values=False, 
                                use_block_cache=True, 
                                block_past_key_values=block_past_key_values, 
                                replace_position=small_block_start_idx,
                                skip_policy=skip_policy if active else None,
                                compute_counter=compute_counter,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            if skip_policy:
                                skip_policy.update_hidden_states(output.hidden_states)
                    else:
                        output = self.forward(
                            input_ids=x_t[:, -block_size:], 
                            use_cache=True, 
                            past_key_values=past_key_values, 
                            update_past_key_values=False,
                            compute_counter=compute_counter,
                        )
                        logits = output.logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]

                    if output_scores:
                        scores_list.append(logits)
                    if output_hidden_states and hasattr(output, 'hidden_states'):
                        decoder_hidden_states.append(output.hidden_states)

                    x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                    x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                    unmask_idx = (x1_p > threshold)
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]

                    if compute_counter:
                        compute_counter.log_unmasked(unmask_idx.sum().item())

                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

        input_ids = x_t
        
    # Truncate stop_token
    if stop_token in input_ids[:, original_input_length:]:
        stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
        input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
    
    if return_dict_in_generate:
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=tuple(scores_list) if output_scores and scores_list else None,
            hidden_states=tuple(decoder_hidden_states) if output_hidden_states and decoder_hidden_states else None,
        )
    else:
        return input_ids
