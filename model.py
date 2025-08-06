import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import transformers
from typing import Callable, List, Optional, Tuple, Union

# GPT-small의 Hyperparameters를 저장하는 설정 클래스
# 1. GQA (Grouped Query Attention) : Query는 여러 개지만 K,V는 더 적은 수로 그룹핑해서 쓰는 MHA방식
#                                    여러 개의 Q가 하나의 K,V를 공유하는 구조이다.
# 2. RoPE: 기존의 위치 임베딩을 대신해, 벡터를 회전시켜 위치 정보를 반영하는 방법
#          기존 방식보다 더 문맥 보존이 잘 되고, 무한 길이 일반화가 잘 됨
# 3. assert문: 주어진 조건이 True가 아니라면 프로그램을 즉시 멈추고 에러를 발생시키는 디버깅 도구이다.
#             hidden size는 attention 수로 나누어 떨어져야 한다는 사전 조건이 있으므로 위반시 중단
class TransformerConfig(transformers.PretrainedConfig):
    model_type = "custom_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4, # GQA를 위한 num_key_value_heads
        head_dim: Optional[int] = None,
        max_postion_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_theta: float = 10000.0, # RoPE의 theta값
        attention_dropout: float = 0.1, # Dropout 확률
        ffn_dropout: float = 0.0, # Dropout 확률
        **kwargs # 키워드 인자를 딕셔너리 형태로 받아오는 방법
    ) -> None:
        super().__init__(**kwargs) # GPT 모델의 설정을 HuggingFace의 PretrainedConfig에 넘겨주기 위해 사용됨
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_postion_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout

        if self.head_dim is None:
            self.head_dim = hidden_size // num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        assert self.hidden_size == (self.num_attention_heads * self.head_dim), "hidden_size must be equal to num_attention_heads * head_dim"

# 위치 정보를 hidden state에 회전 방식으로 반영
# RoPE는 입력 벡터를 아래처럼 절반으로 나누고, 각 부분에 회전 변환을 적용한다.
def apply_rotary_emb(x, position_embeddings):
    cos,sin = position_embeddings # 둘 다 (B,L,D/2)

# x의 차원은 [16,16,1024,32] 즉, [B,H,L,D] 구조이다.
    x1 = x[..., ::2]  # 짝수 차원 / (B,H,L,D/2)
    x2 = x[..., 1::2] # 홀수 차원 / (B,H,L,D/2)

    # cos, sin reshape: (B, 1, D/2, L) -> (B, 1, L, D/2) 으로 바꿔야함
    cos = cos.unsqueeze(1).transpose(2,3)
    sin = sin.unsqueeze(1).transpose(2,3)

    # 회전 적용
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    # 다시 원래 head_dim으로 합치기
    x = x_rotated.flatten(-2)  # 마지막 두 차원 붙이기

    return x

# K,V의 head 수를 늘려서 Query와 일치시키는 GQA 구조를 구현하는 함수이다.
# Query는 보통 head 수가 많고, K,V는 그룹핑되어 적은 수만 존재하므로 K,V를 Q의 개수만큼 복제(Repeat) 해야함.
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: # 복제할 필요가 없다면 그대로 반환
        return hidden_states
    # 복제 해야한다면 중간에 None을 넣어서 차원 하나 늘리고 n_rep 개수만큼 expand로 복제
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# 평균 없이 Root Mean Square만으로 정규화 하는 방법이다.
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)  
        rms = norm / (x.shape[-1] ** 0.5)     
        output = x / (rms + self.eps) * self.weight
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

# inv_freq(=inverse frequency): 각 차원의 고유 주파수
# 위치 정보를 벡터에 회전 형태로 반영하기 위해 각 차원마다 서로 다른 주파수를 부여한다.
class RotaryEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.config = config

        head_dim = config.head_dim
        theta = config.rope_theta
        
        # RoPE inverse frequency 계산
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = inv_freq.unsqueeze(0)  # (1, head_dim // 2)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

# 각 위치마다 회전에 사용할 cos, sin을 생성
    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.view(-1)  # (1, 16) → (16,)

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # .autocast는 어떤 '상태'를 설정하는 것 -> 앞으로 이 블록 안에서 실행되는 연산은 float32를 auto casting한다.
        with torch.autocast(device_type=x.device.type, enabled=False): # disable autocasting for fp32 precision
            # 각도 계산
            angle = position_ids_expanded * inv_freq_expanded
            # cos, sin 값 계산
            cos = torch.cos(angle)  # (batch, seq_len, head_dim // 2)
            sin = torch.sin(angle) 
            # cos,sin만 구하면 apply_rotary_emb에서 rotated를 계산한다.
        return cos.to(x.dtype), sin.to(x.dtype)

class MultiHeadAttention(nn.Module):
# B: batch 크기, T: 시퀀스 길이, H: attention head 개수, G: key-value head 개수 (GQA), D: head 한 개의 차원(head_dim), E = H x D (hidden_state)
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.num_heads = config.num_attention_heads  # 16
        self.head_dim = config.hidden_size // config.num_attention_heads # 512 / 16 = 32
        
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # 16 / 4 = 4
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        # hidden_size: int = 512
        self.q_proj = nn.Linear(config.hidden_size,config.hidden_size, bias=False)                
        kv_out_dim = config.num_key_value_heads * self.head_dim # 4 * 32 = 128
        self.k_proj = nn.Linear(config.hidden_size, kv_out_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_out_dim, bias=False)    

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[List[Tuple[torch.Tensor,torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.size() # hidden_states.shape = (B, T, H * D)

        # 원하는 차원 = (batch , num_heads, seq_len, head_dim)
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        # 현재는 (batch , seq_len, num_heads, head_dim) 이므로 2,3번째의 위치를 바꾼다.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (16, 16, seq_len, 32)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (16, 4, seq_len, 32)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (16, 4, seq_len, 32)

        # RoPE 적용
        query_states = apply_rotary_emb(query_states, position_embeddings)
        key_states = apply_rotary_emb(key_states, position_embeddings)

        # 이전 캐시를 저장하는 것 -> Auto-Regressive 모델에서는 이전에 계산했던 Q,K,V를 다시 계산하면 비효율적이기에 이전의 K,V를 캐싱해둔다.
        if past_key_value is not None:
            key_cache, value_cache = past_key_value
            key_states = torch.cat([key_cache, key_states], dim=-2)
            value_states = torch.cat([value_cache, value_states], dim=-2)
            past_key_value = (key_states, value_states)

        # GQA용 K,V 4배 복제
        key_states = repeat_kv(key_states, self.num_key_value_groups) # (16, 16, seq_len, 32)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # (16, 16, seq_len, 32)

        # 1. Attention Score
        # query_states.shape = (16, 16, seq_len, 32) / key_states.transpose(-2, -1).shape = (16, 16, 32, seq_len)
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # 2. Attention Mask
        if attention_mask is not None:
            if attention_mask.size(-1) < attn_scores.size(-1):
                pad_len = attn_scores.size(-1) - attention_mask.size(-1)
                pad = torch.zeros_like(attention_mask[..., :1].expand(*attention_mask.shape[:-1], pad_len))
                attention_mask = torch.cat([attention_mask, pad], dim=-1)
            attn_scores = attn_scores + attention_mask

        # 3. Softmax, Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 4. Weighted Sum
        # attn_weights.shape: (16,16,seq_len,seq_len) / value_states.shape = (16,16,seq_len,32) -> (16,16,seq_len,32)
        attn_output = torch.matmul(attn_weights, value_states)

        # 5. Concatenate heads
        # attn_output.transpose(1, 2).shape = (16, seq_len, 16, 32) -> (16, seq_len, 512)
        attn_output = (attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim))

        # 6. Output Projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output, past_key_value

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,bias=False)
        self.act_fn = torch.nn.SiLU()
        self.dropout = nn.Dropout(config.ffn_dropout)

    # FFN(x) = Down(SiLU(Gate(x)) * Up(x))
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x)) # Gate(x) = x @ W_gate
        up = self.up_proj(x)                  # Up(x)   = x @ W_up   
        out = gate * up                          
        out = self.down_proj(out)             # Down(.) = @ W_down
        out = self.dropout(out)
        return out

# 트랜스포머 layer 구현
class TransformerLayer(nn.Module): 
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config)

        self.feed_forward = FeedForwardNetwork(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # 1. Layernorm
        normed_hidden = self.input_layernorm(hidden_states)

        # 2. Self-attention
        attn_output, past_key_value = self.self_attn(
            normed_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )

        # 3. Residual connection
        hidden_states = hidden_states + attn_output

        # 4. LayerNorm before FFN
        normed_hidden = self.post_attention_layernorm(hidden_states)

        # 5. FeedForward
        ff_output = self.feed_forward(normed_hidden)

        # 6. Residual connection
        hidden_states = hidden_states + ff_output

        return hidden_states, past_key_value

class TransformerPreTrainedModel(transformers.PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TransformerModel(TransformerPreTrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
    ):
        batch_size, seq_len = input_ids.shape
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        if inputs_embeds is None and input_ids is None:
            raise ValueError("You have to specify either input_ids or input_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = inputs_embeds

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, seq_len)
            
        target_length = seq_len
        seen_token_length = 0
        if past_key_values is not None:
            seen_token_length = past_key_values[0][0].shape[-2]
            target_length += seen_token_length
        
        attention_mask = self._prepare_attention_mask(
            attention_mask=attention_mask,
            sequence_length=seq_len,
            target_length=target_length,
            seen_token_length=seen_token_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            batch_size=batch_size,
        )

        hidden_states = inputs_embeds
        position_embed = self.rotary_emb(hidden_states, position_ids)
        kv_cache_new = []
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values[layer_idx] if past_key_values is not None else None,
                    position_embed
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                    position_embeddings=position_embed,
                )

            hidden_states, kv_cache = layer_outputs
            kv_cache_new.append(kv_cache)

        hidden_states = self.norm(hidden_states)

        if past_key_values is not None:
            past_key_values = kv_cache_new

        return hidden_states, past_key_values

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        seen_token_length: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int
    ):
        # 1. Causal mask 
        causal_mask = torch.tril(torch.ones((target_length, target_length), dtype=dtype, device=device))
        causal_mask = causal_mask[-sequence_length:, :]  # 현재 토큰 기준

        # (1, 1, seq_len, target_len) → 모든 head에 적용 가능
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, sequence_length, target_length)

        # 2. Padding mask
        if attention_mask is not None:
            # (B, seq_len) → (B, 1, seq_len, 1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            causal_mask = causal_mask * attention_mask

        # 3. 마스킹된 위치는 -inf, 나머지는 0 (logits에 더할 것이므로)
        mask = (1.0 - causal_mask) * torch.finfo(dtype).min
        return mask

class TransformerForCausalLM(TransformerPreTrainedModel):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ):  
        if use_cache and past_key_values is None:
            batch_size, _ = input_ids.shape
            dummy = torch.empty((batch_size, self.config.num_key_value_heads, 0, self.config.head_dim)).to(self.model.layers[0].self_attn.q_proj.weight)
            past_key_values = [(dummy.clone(),dummy.clone()) for _ in range(self.config.num_hidden_layers)]
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        # CE Loss를 사용하여 (B, L, vocab_size)의 logits와 (B, L)의 labels를 비교하여 스칼라 loss값을 얻는 것
        logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            logits = logits.float()  # cast to fp32 for calculating softmax in high precision
            shift_logits  = logits[..., :-1, :].contiguous()   # (B, L-1, V)
            shift_labels  = labels[..., 1:].contiguous()       # (B, L-1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size),shift_labels.view(-1))

        return (loss,logits) if loss is not None else (logits, past_key_values)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        max_new_tokens: int = 32,
        return_response_only: bool = False,
    ):
        batch_size, init_seq_len = input_ids.shape
        device = input_ids.device
        eos = self.config.eos_token_id

        unfinish_flag = torch.ones(batch_size, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, -1)
        
        for _ in range(max_new_tokens):
            logits, past_key_values = self.forward(
                input_ids=input_ids[:, -1:] if past_key_values is not None else input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids[:, -1:] if past_key_values is not None else position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinish_flag + eos * (1 - unfinish_flag)
            unfinish_flag = unfinish_flag * next_tokens.ne(eos)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)], dim=-1)

            if unfinish_flag.sum() == 0:
                break
        if return_response_only:
            return input_ids[:, init_seq_len:]
        return input_ids

class TransformerForSequenceClassification(TransformerPreTrainedModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = TransformerModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor=None,
        inputs_embeds: torch.FloatTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        logits = self.classifier(hidden_states[:, -1, :])
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), reduction="mean")
        return (loss, logits) if loss is not None else (logits, past_key_values)