from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaModel,
)


class ContextClassifier(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.n_features = (
            len(config.tokens_to_encode_ids) + 1
        )  # 4 special tokens +1 for the pooled hidden states
        self.score_hidden_size = config.hidden_size * self.n_features
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)

        self.hidden = nn.Linear(
            self.score_hidden_size,
            self.score_hidden_size,
            bias=True,
        )

        self.activation = nn.ReLU()

        self.score = nn.Linear(
            self.score_hidden_size,
            self.num_labels,
            bias=True,
        )

        self.tokens_to_encode_ids = config.tokens_to_encode_ids

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.last_hidden_state
        batch_size, _, hidden_size = hidden_states.shape

        # get the hidden states of the special tokens - vectorized version
        hidden = torch.zeros(
            (batch_size, self.n_features - 1, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Create a mask for each special token (batch_size, seq_len, n_special_tokens)
        token_masks = torch.stack(
            [input_ids == token_id for token_id in self.tokens_to_encode_ids], dim=-1
        )

        # Use masked_select and reshape to gather all matching hidden states
        # This avoids the Python-level loop and does everything in CUDA
        for token_idx in range(len(self.tokens_to_encode_ids)):
            mask = token_masks[..., token_idx]
            if mask.any():
                # Get the first occurrence of each special token
                first_occurs = mask.float().nonzero()
                # Gather the hidden states for these positions
                hidden[first_occurs[:, 0], token_idx] = hidden_states[
                    first_occurs[:, 0], first_occurs[:, 1]
                ]

        hidden = hidden.view(batch_size, -1)

        # get a context representation by doing max pooling over the hidden states
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        # mean pool the hidden states of all tokens ignoring the mask tokens
        mask = (
            attention_mask.unsqueeze(-1)
            .expand(hidden_states.size())
            .to(hidden_states.dtype)
        )
        masked_hidden_states = hidden_states * mask
        sum_hidden_states = torch.sum(masked_hidden_states, dim=1)
        sum_attention_mask = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_hidden_states = sum_hidden_states / sum_attention_mask
        pooled_hidden_states = pooled_hidden_states.to(hidden.device)

        hidden = torch.cat([hidden, pooled_hidden_states], dim=1)
        hidden = self.hidden(hidden)
        hidden = self.activation(hidden)
        logits = self.score(hidden)

        return logits
