#

# wrapper of bart to support input_embs
# note: mostly from "transformers.modeling_bart" (v3.1.0)

from transformers import BartConfig, BartModel
from transformers.modeling_bart import BartEncoder, BartDecoder

# --
class MyBartEncoder(BartEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None,
                output_attentions=False, output_hidden_states=False, return_dict=False):
        # --
        from transformers.modeling_bart import invert_mask, F, random, BaseModelOutput
        # --
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        # --
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        else:
            inputs_embeds = inputs_embeds * self.embed_scale
        # --
        # embed_pos = self.embed_positions(input_ids)
        embed_pos = self.embed_positions(inputs_embeds)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # --
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # --
        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attn,)
        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)
        # --

# --
class MyBartDecoder(BartDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_causal_val(self, input_length: int, past_length: int, device):
        import torch
        full_length = past_length + input_length
        a1 = (torch.arange(input_length) + past_length).to(device)  # [Curr]
        a2 = torch.arange(full_length).to(device)  # [Past + Curr]
        ret = torch.zeros((input_length, full_length)).float().to(device)  # [Curr, Past+Curr]
        ret[a1.unsqueeze(-1) < a2.unsqueeze(-2)] = float('-inf')
        return ret

    def forward(self, input_ids, attention_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=False,
                output_attentions=False, output_hidden_states=False, return_dict=False):
        # --
        from transformers.modeling_bart import invert_mask, F, torch, BaseModelOutputWithPast, random
        # --
        # check attention mask and invert
        encoder_padding_mask = invert_mask(encoder_attention_mask) if encoder_attention_mask is not None else None
        decoder_padding_mask = invert_mask(attention_mask) if attention_mask is not None else None
        # embed positions
        _prev_key = None if past_key_values is None else past_key_values[0].get('self',{}).get('prev_key',None)
        if _prev_key is None:
            past_length = 0
        else:
            past_length = _prev_key.size(-2)
        input_length = inputs_embeds.size(-2) if inputs_embeds is not None else input_ids.size(-1)
        input_device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
        _offset = getattr(self.embed_positions, 'offset', 0)  # note: a specific hack ...
        position_ids = (_offset+past_length) + torch.arange(input_length, dtype=torch.long, device=input_device)
        positions = super(type(self.embed_positions), self.embed_positions).forward(position_ids)
        # causual
        _causal_val = self.prepare_causal_val(input_length, past_length, input_device)
        # embeddings
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids) * self.embed_scale
        else:
            x = inputs_embeds * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Convert to Bart format: (BS, seq_len, model_dim) -> (seq_len, BS, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_state = past_key_values[idx] if past_key_values is not None else None
            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=_causal_val,
                output_attentions=output_attentions,
            )
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if self.layer_norm and (idx == len(self.layers) - 1):  # if config.add_final_layer_norm (mBART)
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)
        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )

    def reorder_cache(self, past, beam_idx):
        from transformers import BartForConditionalGeneration
        return BartForConditionalGeneration._reorder_cache(past, beam_idx)

# --
class MyBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # --
        from torch import nn
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = MyBartEncoder(config, self.shared)  # note: change to MyBart
        self.decoder = MyBartDecoder(config, self.shared)
        self.init_weights()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self, *args, **kwargs):
        raise RuntimeError("No directly calling of this!!")

    @classmethod
    def from_config(cls, config):
        return cls(config)
