from typing import *
import numpy as np
import torch.nn as nn
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
import torch


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        # attention layer
        self.multihead_attn = nn.MultiheadAttention(768, 8, batch_first= True)

    # change return_dict from True to False
    def forward(
        self,
        hidden_states: torch.Tensor = None,
        hidden_states2: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask2: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        head_mask2: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_hidden_states2: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask2: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        alpha=1000,
        mix_layer=1000,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        lbeta = -1

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        if mix_layer == -1 and hidden_states2 is not None:
            attn_hidden_states2, attn_output_weights2 = self.multihead_attn(hidden_states2, hidden_states2, hidden_states2)
                    
            attn = torch.sum(attn_hidden_states2, dim = 2)
                    
            lbeta = 0
            while lbeta < 0.5:
                theta = np.random.beta (alpha, alpha)
                shreshold = (theta * torch.max(attn, dim = 1).values).unsqueeze(-1)
                shreshold = torch.cat([shreshold] * attn.shape[1], dim = 1)
                mask = attn.gt(shreshold).unsqueeze(-1)
                mask = torch.cat([mask] * attn_hidden_states2.shape[-1], dim=-1)
                lbeta = 1- hidden_states[mask].shape[0] / (hidden_states.shape[0] * hidden_states.shape[1] * hidden_states.shape[2])
            hidden_states[mask] = hidden_states2[mask]

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if hidden_states2 is not None:
                    layer_head_mask2 = head_mask2[i] if head_mask2 is not None else None

                if self.gradient_checkpointing and self.training:

                    if use_cache:
                        # logger.warning(
                        #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        # )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
                    if hidden_states2 is not None:
                        layer_outputs2 = layer_module(
                            hidden_states2,
                            attention_mask2,
                            layer_head_mask2,
                            encoder_hidden_states2,
                            encoder_attention_mask2,
                            past_key_value,
                            output_attentions,
                        )

                hidden_states = layer_outputs[0]
                if hidden_states2 is not None:
                    hidden_states2 = layer_outputs2[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            if i == mix_layer:
                # tmix
                if hidden_states2 is not None:
                    attn_hidden_states2, attn_output_weights2 = self.multihead_attn(hidden_states2, hidden_states2, hidden_states2)
                    
                    attn = torch.sum(attn_hidden_states2, dim = 2)
                    
                    lbeta = 0
                    while lbeta < 0.5:
                        theta = np.random.beta (alpha, alpha)
                        shreshold = (theta * torch.max(attn, dim = 1).values).unsqueeze(-1)
                        shreshold = torch.cat([shreshold] * attn.shape[1], dim = 1)
                        mask = attn.gt(shreshold).unsqueeze(-1)
                        mask = torch.cat([mask] * attn_hidden_states2.shape[-1], dim=-1)
                        lbeta = 1- hidden_states[mask].shape[0] / (hidden_states.shape[0] * hidden_states.shape[1] * hidden_states.shape[2])
                    hidden_states[mask] = hidden_states2[mask]


            if i > mix_layer:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:

                    if use_cache:
                        # logger.warning(
                        #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        # )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    lbeta,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertModel4Mix(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    # change return_dict from None to False
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_ids2: Optional[torch.Tensor] = None,
        lbeta = 1000,
        mix_layer = 1000,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_ids2: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        head_mask2: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        inputs_embeds2: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states2: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        # suppose the bool values in config are False

        # config.output_attentions is a bool, True or False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            if input_ids2 is not None or inputs_embeds2 is not None:
                attention_mask2 = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
                if input_ids2 is not None or inputs_embeds2 is not None:
                    buffered_token_type_ids2 = self.embeddings.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded2 = buffered_token_type_ids2.expand(batch_size, seq_length)
                    token_type_ids2 = buffered_token_type_ids_expanded2
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                if input_ids2 is not None or inputs_embeds2 is not None:
                    token_type_ids2 = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        if input_ids2 is not None or inputs_embeds2 is not None:
            extended_attention_mask2: torch.Tensor = self.get_extended_attention_mask(attention_mask2, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                if input_ids2 is not None or inputs_embeds2 is not None:
                    encoder_attention_mask2 = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            if input_ids2 is not None or inputs_embeds2 is not None:
                encoder_extended_attention_mask2 = self.invert_attention_mask(encoder_attention_mask2)
        else:
            encoder_extended_attention_mask = None
            if input_ids2 is not None or inputs_embeds2 is not None:
                encoder_extended_attention_mask2 = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        if input_ids2 is not None or inputs_embeds2 is not None:
            head_mask2 = self.get_head_mask(head_mask2, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        if input_ids2 is not None or inputs_embeds2 is not None:
            embedding_output2 = self.embeddings(
                input_ids=input_ids2,
                position_ids=position_ids2,
                token_type_ids=token_type_ids2,
                inputs_embeds=inputs_embeds2,
                past_key_values_length=past_key_values_length,
            )

            encoder_outputs = self.encoder(
                hidden_states=embedding_output,
                hidden_states2=embedding_output2,
                attention_mask=extended_attention_mask,
                attention_mask2=extended_attention_mask2,
                head_mask=head_mask,
                head_mask2=head_mask2,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states2=encoder_hidden_states2,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_attention_mask2=encoder_extended_attention_mask2,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                alpha=lbeta,
                mix_layer=mix_layer,
            )
        else:
            encoder_outputs = self.encoder(
                hidden_states=embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = encoder_outputs[0]
        lbeta = encoder_outputs[1]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, lbeta) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class MixText(nn.Module):
    def __init__(self, num_labels = 2, mix_option=False, model='bert-base-chinese'):
        """
        Mix_Text Model:
        mix or not bert + sequential linear
        :param num_labels: number of classification label
        :param mix_option: mix_option=True, using mixText
                           mix_option=False, using Bert
        :param model:
        """
        super(MixText, self).__init__()
        if mix_option:
            self.bert = BertModel4Mix.from_pretrained(model)
        else:
            self.bert = BertModel.from_pretrained(model)

        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, input_ids = None, input_ids2 = None, inputs_embeds = None, inputs_embeds2 = None, lbeta=None, mix_layer=1000):
        """

        :param x:
        :param x2:  x2 in <forward>func is consistent with mix_option in <__init__>func
                    mix_option is False while x2 is None;
                    mix_option is True while x2 is not None;
        :param lbeta:
        :param mix_layer:
        :return:
        """

        if input_ids2 is not None or inputs_embeds2 is not None:
            # this bert is mixText-Bert Model, just changing Encoder in Bert to Encoder4Mix
            # <torch.mean> 1 means that the dim of 1 is reduced
            outputs = self.bert(input_ids=input_ids, input_ids2=input_ids2, inputs_embeds=inputs_embeds, inputs_embeds2=inputs_embeds2, lbeta=lbeta, mix_layer=mix_layer)
            pooled_output = torch.mean(outputs[0], 1)
        else:
            outputs = self.bert(input_ids=input_ids)
            pooled_output = torch.mean(outputs[0], 1)

        # outputs[0] : [batch_size, ?, embedding_dim] , which ? is reduced
        # pooled_output : [batch_size, embedding_dim]
        # predict : [batch_size, num_labels]
        predict = self.linear(pooled_output)
        return predict, outputs[2]
