import torch.nn as nn
import torch
from torchinfo import summary
import math

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)


        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        ) 

        attn_score = torch.matmul(query, key) / math.sqrt(self.head_dim)


        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)  
        attn_score = torch.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_score, value)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
   
        residual = x
        out = self.attn(x, x, x)  
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class MemoryRecallLayer(nn.Module):
    def __init__(self, query_dim, memory_dim, num_heads):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads

        if query_dim == memory_dim:
            self.query_proj = nn.Identity()
        else:
            self.query_proj = nn.Linear(query_dim, memory_dim)
        self.cross_attn = AttentionLayer(memory_dim, num_heads=num_heads) 

    def forward(self, query_states, memory_key_value_states):
        batch_size, L_q, _ = query_states.shape
        batch_size_m, L_m, _ = memory_key_value_states.shape
        assert batch_size == batch_size_m, "Batch sizes must match"

        projected_query = self.query_proj(query_states) 

        recalled_info = self.cross_attn(projected_query, memory_key_value_states, memory_key_value_states) 

        return recalled_info


class HSM_ChannelMixerLayer(nn.Module):
    def __init__(self, dim, in_steps, num_nodes, state_dim=64, d_inner_ratio=2.0, A_init_range=(1, 16), use_1d_conv: bool = False):
        super().__init__()
        self.dim = dim
        self.in_steps = in_steps
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.d_inner_ratio = d_inner_ratio
        self.A_init_range = A_init_range
        self.use_1d_conv = use_1d_conv

        self.d_inner = int(dim * d_inner_ratio)

        self.norm_in = nn.LayerNorm(dim)
        self.BCdt_proj = nn.Linear(dim, 3 * state_dim)
        
        self.act = nn.SiLU()

        if self.use_1d_conv:
            self.dw_conv1d = nn.Conv1d(
                in_channels=3 * self.state_dim,
                out_channels=3 * self.state_dim,
                kernel_size=3,
                padding=2,
                groups=3 * self.state_dim,
                bias=False
            )
        else:
            self.dw_conv1d = None

        self.A_param = nn.Parameter(torch.empty(state_dim))
        nn.init.uniform_(self.A_param, *A_init_range)

        self.hz_proj = nn.Linear(dim, 2 * self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)

        self.D_param = nn.Parameter(torch.ones(dim))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        B_batch, _, _, D_model = x.shape
        L_seq = self.in_steps * self.num_nodes

        x_flat = x.reshape(B_batch, L_seq, D_model)
        x_norm = self.norm_in(x_flat)

        bcd_intermediate = self.BCdt_proj(x_norm)
        bcd_intermediate_perm = bcd_intermediate.transpose(1, 2)

        processed_bcd: torch.Tensor
        if self.use_1d_conv and self.dw_conv1d is not None:
            bcd_conv = self.dw_conv1d(bcd_intermediate_perm)
            processed_bcd = bcd_conv[..., :L_seq]
        else:
            processed_bcd = bcd_intermediate_perm
        
        B_s, C_s, dt_s = torch.split(processed_bcd, [self.state_dim, self.state_dim, self.state_dim], dim=1)

        A_s = (dt_s + self.A_param.view(1, -1, 1)).softmax(dim=-1)

        x_norm_perm = x_norm.transpose(1, 2)
        AB_s = A_s * B_s
        h_compressed = x_norm_perm @ AB_s.transpose(-1, -2)

        h_compressed_perm = h_compressed.transpose(1, 2)
        h_split, z_split = torch.split(self.hz_proj(h_compressed_perm), [self.d_inner, self.d_inner], dim=-1)
        h_mixed_channel = self.out_proj(h_split * self.act(z_split))

        y_intermediate = h_mixed_channel.transpose(1, 2) @ C_s

        y_perm = y_intermediate + x_norm_perm * self.D_param.view(1, -1, 1)
        y = y_perm.transpose(1, 2)
        y_norm_out = self.norm_out(y)

        out = y_norm_out.reshape(B_batch, self.in_steps, self.num_nodes, D_model)
        return out


class RecallMamba(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        t_num_layers=1,
        s_num_layers=1,
        dropout=0.1,
        use_mixed_proj=True,
        hsm_state_dim=64, 
        hsm_d_inner_ratio=2.0, 
        hsm_num_layers=1,     
        memory_window_size=8,
        max_memory_items_for_recall=4, 
        hsm_use_1d_conv: bool = False, 
        **kwargs
    ):
        super().__init__()

        self.memory_window_size = memory_window_size
        self.max_memory_items_for_recall = max_memory_items_for_recall 
        self.memory_bank = [] 



        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.t_num_layers = t_num_layers
        self.s_num_layers = s_num_layers
        self.use_mixed_proj = use_mixed_proj
        self.hsm_num_layers = hsm_num_layers 
        self.hsm_use_1d_conv = hsm_use_1d_conv 
 
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.input_embeding = TokenEmbedding(input_dim, input_embedding_dim)

        self.memory_recall_layer = MemoryRecallLayer(self.model_dim, self.model_dim, self.num_heads)
        self.fusion_gate_linear = nn.Linear(self.model_dim * 2, self.model_dim)     


        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        self.dummy_rep_token = torch.nn.Parameter(torch.randn(in_steps, num_nodes, adaptive_embedding_dim))

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(t_num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(s_num_layers)
            ]
        )


        self.hsm_layers = nn.ModuleList(
            [
                HSM_ChannelMixerLayer(
                    dim=self.model_dim,
                    in_steps=self.in_steps,
                    num_nodes=self.num_nodes,
                    state_dim=hsm_state_dim,
                    d_inner_ratio=hsm_d_inner_ratio,
                    use_1d_conv=self.hsm_use_1d_conv 
                )
                for _ in range(self.hsm_num_layers)
            ]
        )

    def forward(self, x):
    

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., :self.input_dim]

        x = self.input_embeding(x)

        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            ) 
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)  

        batch_size, in_steps, num_nodes, model_dim = x.shape

        x_flat_for_memory = x.reshape(batch_size, in_steps * num_nodes, model_dim)
        self.memory_bank.append(x_flat_for_memory.detach())
        if len(self.memory_bank) > self.memory_window_size:
            self.memory_bank.pop(0) 

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        current_batch_size, current_in_steps, current_num_nodes, current_model_dim = x.shape

        x_query_for_recall = torch.mean(x, dim=(1, 2), keepdim=True) 
        x_query_for_recall = x_query_for_recall.squeeze(2) 


        memory_key_value_states_from_bank = None
        if self.memory_bank:
            compatible_memories = [
                mem for mem in self.memory_bank
                if mem.shape[0] == current_batch_size and mem.shape[2] == current_model_dim
            ]

            if compatible_memories:
                relevant_memories = compatible_memories[-self.max_memory_items_for_recall:]
                try:
                    memory_key_value_states_from_bank = torch.cat(relevant_memories, dim=1)
                except RuntimeError as e:
                    print(f"Error concatenating relevant memory bank: {e}. Proceeding without memory recall for this batch.")
                    memory_key_value_states_from_bank = None
            else:
                pass

        fused_output = x 

        if memory_key_value_states_from_bank is not None and memory_key_value_states_from_bank.nelement() > 0 :

                 recalled_info_compressed = self.memory_recall_layer(x_query_for_recall, memory_key_value_states_from_bank)
                 recalled_info_expanded = recalled_info_compressed.unsqueeze(2).expand(-1, current_in_steps, current_num_nodes, -1)


                 gate_input = torch.cat((x, recalled_info_expanded), dim=-1)
                 gate = torch.sigmoid(self.fusion_gate_linear(gate_input)) 
                 fused_output = gate * x + (1 - gate) * recalled_info_expanded

  
        x = fused_output

        for layer in self.hsm_layers:
            x = layer(x)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  
        else:
            out = x.transpose(1, 3)  
            out = self.temporal_proj(
                out
            ) 
            out = self.output_proj(
                out.transpose(1, 3)
            )  

        return out


if __name__ == "__main__":
    model = RecallMamba(207, 12, 12)
    summary(model, [64, 12, 207, 3])
