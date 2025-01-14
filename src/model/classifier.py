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


class Classifier(LlamaForSequenceClassification):
    def __init__(self, config):
        config.loss_type = "ForSequenceClassification"
        super().__init__(config)

        self.num_special_tokens = 4  # the number of special tokens per sequence
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(
            config.hidden_size * self.num_special_tokens, self.num_labels, bias=True
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

        batch_size, seq_len = input_ids.shape

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
        hidden_states = (
            transformer_outputs.last_hidden_state
        )  # (batch_size, seq_len, hidden_size)

        # get a indexes of the special tokens in the input_ids
        special_tkns_idxs = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=hidden_states.device
        )
        for token_id in self.tokens_to_encode_ids:
            special_tkns_idxs += input_ids == token_id

        special_tkns_hidden_state = hidden_states[special_tkns_idxs]
        special_tkns_hidden_state = special_tkns_hidden_state.view(batch_size, -1)

        logits = self.score(special_tkns_hidden_state)

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                pooled_logits=logits,
                config=self.config,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
