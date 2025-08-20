import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLForConditionalGeneration
from V2P.constants import IGNORE_INDEX
from typing import List, Tuple, Union, Optional
from V2P.trainer import rank0_print

target_dist_stats = {'total_count': 0, 'total_nonzero_count': 0.0, 'total_nonzero_max': 0.0, 'total_nonzero_min_sum':0.0}


class QwenVLwithVisionHeadOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output class for Qwen2_5_VL with pointer head, extending the base output class.
    
    Args:
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        pointer_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vision pointer network loss.
        pointer_scores (`List[torch.FloatTensor]`, *optional*):
            Attention scores from the pointer network, one tensor per batch item.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (weighted sum of lm_loss and pointer_loss).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores from the language modeling head.
        past_key_values, hidden_states, attentions, rope_deltas:
            Same as parent class.
    """
    def __init__(self, lm_loss=None, pointer_loss=None, pointer_scores=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_loss = lm_loss
        self.pointer_loss = pointer_loss
        self.pointer_scores = pointer_scores
        self.forward_kl_weight = kwargs.get("forward_kl_weight", 1.0)
        self.reverse_kl_weight = kwargs.get("reverse_kl_weight", 0.0)


class VisionHead_MultiPatch(nn.Module):
    def __init__(self, d_model, projection_dim, num_attention_heads=8, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Note: We omit additional normalization here because Qwen2VL
        # already normalizes hidden states using RMSNorm.
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model)
        )

        # Add self-attention layer for visual features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization and residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                hidden_state_enc,  # shape: [n_enc, d_model] where n_enc can vary with image size
                hidden_state_dec,  # shape: [n_dec, d_model] there can be multiple query in one sample
                labels: Optional[torch.Tensor] = None,  # shape: [n_dec, n_enc], binary mask of patches in bbox
                do_single_patch: bool = False,
               ):
        
        enc_input = hidden_state_enc.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input,
            key=enc_input,
            value=enc_input,
            # attn_mask=attention_mask,
            need_weights=False
        )
        # Residual connection and layer normalization
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        # Remove batch dimension
        hidden_state_enc_ctx = hidden_state_enc_ctx.squeeze(0)  # [n_enc, d_model]

        # Apply the projection networks.
        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)  # [n_dec, d_model]
        # print(f"{proj_enc.size()=}")
        # print(f"{proj_dec.size()=}")

        # Compute scaled dot-product attention scores.
        # Scaling by sqrt(d_model) is critical regardless of variable n_enc.
        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]
        # print(f"{patch_logits.size()=}")

        # Softmax normalization is applied along the encoder dimension.
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        # reverse_kl_loss = None
        if (labels is not None) and (not do_single_patch):
            epsilon = 1e-8
            labels_float = labels.float()
            # Normalize each row to get target probability distribution
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)
            
            ## add reverse loss
            reversed_labels = 1 - labels  # soft weight, [N, M], optional
            # === 构建 mask：仅保留 label 精确等于 0 的位置 ===
            mask = (labels == 0).float()  # [N, M], 1 when label==0, else 0

            # Apply reversed_labels (or you can skip this and just use mask)
            negative_attention = attn_weights * reversed_labels  # [N, M], soft weighting
            negative_attention_masked = negative_attention * mask  # 零出 label > 0 的位置

            print(f"negative_attention: {negative_attention_masked}")

            # Sum over last dimension
            negative_attn_sum = negative_attention_masked.sum(dim=-1)  # [N]
            print(f"negative_attn_sum: {negative_attn_sum.shape}")
            negative_loss = negative_attn_sum.mean()
            print(f"negative_loss: {negative_loss}")

            # # view
            # import torch.distributed as dist
            # global target_dist_stats
            # if dist.is_available() and dist.is_initialized():
            #     rank = dist.get_rank()
            # else:
            #     rank = 0

            # if rank == 0:
            #     print(f"target_dist size :{target_dist.size()}")
            #     target_dist_stats['total_count'] += 1
            #     nonzero_mask = target_dist > 1e-6
            #     nonzero_elements = target_dist[nonzero_mask]
            #     current_count = nonzero_elements.numel()
                
            #     current_max = nonzero_elements.max().item() if current_count > 0 else 0.0
            #     current_min = nonzero_elements.min().item() if current_count > 0 else float('inf')  # 用 inf 表示无有效最小值

            #     # 更新累计统计
            #     target_dist_stats['total_nonzero_count'] += current_count
            #     target_dist_stats['total_nonzero_max'] += current_max
            #     target_dist_stats['total_nonzero_min_sum'] += current_min  # 累加最小值用于求平均

            #     # 计算平均值
            #     avg_count = target_dist_stats['total_nonzero_count'] / target_dist_stats['total_count']
            #     avg_max = target_dist_stats['total_nonzero_max'] / target_dist_stats['total_count']
            #     avg_min = (target_dist_stats['total_nonzero_min_sum'] / target_dist_stats['total_count']) 

            #     # 处理 avg_min 中可能存在的 inf（比如前几次迭代没有非零元素）
            #     if avg_min == float('inf'):
            #         avg_min = 0.0

            #     print(f"[Rank0] NonZero: {current_count}, Max: {current_max:.6f}, Min: {current_min:.6f}, "
            #         f"AvgNonZero: {avg_count:.6f}, AvgMax: {avg_max:.6f}, AvgMin: {avg_min:.6f}")
            # Apply log_softmax to logits
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            # Use KL divergence as loss
            loss = F.kl_div(pred_log_probs, target_dist, reduction='batchmean')

            # add reverse loss
            loss = loss + negative_loss

            # # Reverse KL
            # pred_dist = pred_log_probs.exp()  # 复用 log-softmax 结果
            # target_log_probs = torch.log(target_dist.clamp(min=1e-8))
            # reverse_kl_loss = F.kl_div(target_log_probs, pred_dist, reduction='batchmean')

            # if rank == 0:
            #     print(f"[Rank0] KL loss:{loss}, Reverse KL loss:{reverse_kl_loss}")

            #     elementwise_kl = target_dist * (torch.log(target_dist + 1e-10) - pred_log_probs)  # 加小数避免 log(0)

            #     # Step 2: 找到每个样本中 target_dist 最大值的位置
            #     max_indices = torch.argmax(target_dist, dim=1)  # (batch_size,)

            #     # Step 3: 提取每个样本中最大值位置对应的 KL 损失贡献
            #     max_contributions = elementwise_kl.gather(1, max_indices.unsqueeze(1))  # (batch_size, 1)

            #     # Step 4: 计算总损失和最大值位置的总贡献
            #     total_loss = elementwise_kl.sum() / elementwise_kl.size(0)  # batchmean: 按 batch 平均
            #     max_contrib_mean = max_contributions.mean()

            #     # 输出
            #     print(f"Total KL loss (batchmean): {total_loss.item():.6f}")
            #     print(f"Contribution from max target_dist positions: {max_contrib_mean.item():.6f}")
            #     print(f"Proportion: {max_contrib_mean / total_loss * 100:.2f}%")

        if do_single_patch and (labels is not None):
            loss = F.cross_entropy(attn_scores, labels)

        # return attn_weights, loss
        return attn_weights, loss


class Qwen2_5_VLForConditionalGenerationWithPointer(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_patch_pointer_head = VisionHead_MultiPatch(self.config.hidden_size, self.config.hidden_size)
        self.pointer_loss_weight = kwargs.get("pointer_loss_weight", 1.0)
        self.lm_loss_weight = kwargs.get("lm_loss_weight", 1.0)
        self.post_init()
    
    def reset_loss_weights(self, pointer_loss_weight, lm_loss_weight):
        self.pointer_loss_weight = pointer_loss_weight
        self.lm_loss_weight = lm_loss_weight
   
    def forward(self,
                input_ids: torch.LongTensor = None, # (batch_size, seq_len)
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                rope_deltas: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                second_per_grid_ts: Optional[torch.Tensor] = None,
                # Grounding
                visual_token_indices_of_coordinates: Optional[torch.Tensor] = None, # shape: (batch_size, n_target); each element is the ground-truth index of the visual token that should be attended to for the corresponding target token
                multi_patch_labels: Optional[torch.Tensor] = None, # shape: list [(n_target, n_visual), ...]; binary mask of patches in bbox
                if_multi_patch: bool = True,
                coordinates: Optional[List[Tuple[float, float]]] = None,
                verbose: bool = False) -> Union[Tuple, QwenVLwithVisionHeadOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f"{multi_patch_labels[0].size()=}")
        if verbose:
            rank0_print(f"input_ids: {input_ids.shape}, {input_ids[0][:5]}...")
            rank0_print(f"labels: {labels.shape}, {labels[0][:5]}...")
            rank0_print(f"pixel_values: {pixel_values.shape}")
            rank0_print(f"image_grid_thw: {image_grid_thw.shape}, {image_grid_thw}")
            rank0_print(f"coordinates: {coordinates}")
            rank0_print(f"visual_token_indices_of_coordinates: {visual_token_indices_of_coordinates}")
            rank0_print(f"return_dict: {return_dict}")

        if inputs_embeds is None:
            # modified by yishan.wd for transformers==4.53.2
            inputs_embeds = self.model.embed_tokens(input_ids) # shape: (batch_size, seq_len, d_model)
            # inputs_embeds = self.model.language_model.embed_tokens(input_ids)
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

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                # modified by yishan.wd for transformers==4.53.2
                # position_ids, rope_deltas = self.model.get_rope_index(
                #     input_ids, image_grid_thw, video_grid_thw, attention_mask
                # )
                # self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] # shape: (batch_size, seq_len, d_model)
        logits = self.lm_head(hidden_states)

        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)


        # If vision supervision is requested, process the action head.
        pointer_loss = None
        pointer_scores = []
        if visual_token_indices_of_coordinates is not None:
            batch_size = input_ids.shape[0]
            pointer_losses = []
            
            # Process each sample individually because the number of visual and target tokens may vary.
            for i in range(batch_size):
                dummy_target = False

                # Get the token ids and corresponding hidden states for sample i.
                token_ids = input_ids[i]          # shape: (seq_length,)
                hs = hidden_states[i]             # shape: (seq_length, d_model)

                # Identify visual tokens indices.
                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1) # shape: (n_visual,)

                # Identify target tokens (the ones that should attend to visual features).
                target_mask = (token_ids == self.config.pointer_pad_token_id)
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
                
                # If either visual or target tokens are missing, skip this sample.
                if visual_indices.numel() == 0:
                    raise ValueError(f"No visual or target tokens found for sample {i}.")
                if target_indices.numel() == 0:
                    target_indices = torch.tensor([hs.shape[0] - 1]) # take the last token as the dummy target token
                    gt = torch.tensor([0]).to(hs.device) # take the first visual token as the dummy ground truth
                    if if_multi_patch:  # task the first 4 visual tokens as the ground truth
                        sample_labels = torch.zeros_like(visual_indices).unsqueeze(0)
                        sample_labels[0][:4] = 1
                        # print(f"0{sample_labels.size()=}")

                        # n_t = target_indices.size(0)          # 目标 token 个数
                        # n_v = visual_indices.size(0)
                        # sample_labels = torch.zeros(
                        #     (n_t, n_v), device=hs.device, dtype=torch.float
                        # )
                        # sample_labels[:, :min(4, n_v)] = 1
                    dummy_target = True
                else:
                    # For supervision, we assume that visual_token_indices_of_coordinates[i] is a tensor of shape (n_target,)
                    # where each element is an integer in the range [0, n_visual-1] indicating the ground-truth visual token.
                    gt = visual_token_indices_of_coordinates[i].to(hs.device) # shape: (n_target,)
                    if if_multi_patch:
                        sample_labels = multi_patch_labels[i]
                        # print(f"1{sample_labels.size()=}")
                        # if sample_labels is None:
                        #     n_t = target_indices.size(0)          # 目标 token 个数
                        #     n_v = visual_indices.size(0)
                        #     sample_labels = torch.zeros(
                        #         (n_t, n_v), device=hs.device, dtype=torch.float
                        #     )
                        #     sample_labels[:, :min(4, n_v)] = 1
                        #     dummy_target = True
                
                # Gather the corresponding hidden state representations.
                # visual_hidden = hs[visual_indices]  # shape: (n_visual, d_model)
                visual_embeds = inputs_embeds[i][visual_indices]
                target_hidden = hs[target_indices]  # shape: (n_target, d_model)

                # Calculate loss for multi-patch mode
                if if_multi_patch:
                    # Ensure the number of targets matches between sample and labels
                    if sample_labels.shape[0] != target_indices.shape[0]:
                        raise ValueError(f"Sample {i} has mismatched target counts: {sample_labels.shape[0]} labels but found {target_indices.shape[0]} target tokens")

                    # Process using VisionHead_MultiPatch
                    # print(f"2{sample_labels.size()=}")

                    attn_scores, loss_v = self.multi_patch_pointer_head(
                        visual_embeds,
                        target_hidden,
                        labels=sample_labels
                    )
                    
                else:
                    # Deprecated branch - single patch mode is no longer used
                    # Run the action head to compute the attention (from target tokens to visual tokens) and its loss.
                    attn_scores, loss_v = self.pointer_head(visual_embeds, target_hidden, labels=gt)
                
                pointer_scores.append(attn_scores.detach().cpu())

                pointer_losses.append(loss_v * 0.0 if dummy_target else loss_v)
            
            pointer_loss = torch.stack(pointer_losses).mean()

        # Combine the LM loss and vision loss using the provided loss weights.
        
        if lm_loss is None:
            total_loss = pointer_loss
        elif pointer_loss is None:
            total_loss = lm_loss
        else:
            total_loss = self.lm_loss_weight * lm_loss + self.pointer_loss_weight * pointer_loss

        if return_dict:
            return QwenVLwithVisionHeadOutputWithPast(
                lm_loss=lm_loss,
                pointer_loss=pointer_loss,
                pointer_scores=pointer_scores,
                loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )
        else:
            # When labels are provided, parent's forward returns a tuple with loss as the first element.
            if labels is not None:
                # Replace the LM loss with the combined loss.
                output = (lm_loss, pointer_loss, logits, pointer_scores,) + outputs[1:]
                # print(f"returning: total_loss, logits, pointer_scores, ...")
                return (total_loss,) + output if total_loss is not None else output
            else:
                return outputs