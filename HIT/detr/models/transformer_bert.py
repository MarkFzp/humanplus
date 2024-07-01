import torch
import torch.nn as nn
#Reference https://github.com/karpathy/nanoGPT

class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate, self_attention=True, query_num=50) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.ln_3 = nn.LayerNorm(latent_dim)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.query_num = query_num
        self.self_attention = self_attention
        
    def forward(self, x):
        if self.self_attention:
            x = self.ln_1(x)
            x2 = self.attn(x, x, x, need_weights=False)[0]
            x = x + self.dropout1(x2)
            x = self.ln_2(x)
            x = x + self.mlp(x)
            x = self.ln_3(x)
            return x
        else:
            x = self.ln_1(x)
            x_action = x[-self.query_num:].clone()
            x_condition = x[:-self.query_num].clone()
            x2 = self.attn(x_action, x_condition, x_condition, need_weights=False)[0]
            x2 = x2 + self.dropout1(x2)
            x2 = self.ln_2(x2)
            x2 = x2 + self.mlp(x2)
            x2 = self.ln_3(x2)
            x = torch.cat((x_condition, x2), dim=0)
            return x
            
    
class Transformer_BERT(nn.Module):
    def __init__(self, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.0,  
                 use_pos_embd_image=False, use_pos_embd_action=False, query_num=50,
                 self_attention=True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.context_len = context_len
        self.use_pos_embd_image = use_pos_embd_image==1
        self.use_pos_embd_action = use_pos_embd_action==1
        self.query_num = query_num
        if use_pos_embd_action and use_pos_embd_image:
            self.weight_pos_embed = None
        elif use_pos_embd_image and not use_pos_embd_action:
            self.weight_pos_embed = nn.Embedding(self.query_num, latent_dim)
        elif not use_pos_embd_image and not use_pos_embd_action:
            self.weight_pos_embed = nn.Embedding(self.context_len, latent_dim)
        elif not use_pos_embd_image and use_pos_embd_action:
            raise ValueError("use_pos_embd_action is not supported")
        else:
            raise ValueError("bug ? is not supported")
        
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate, self_attention, query_num) for _ in range(num_layer)],
        )
        self.self_attention = self_attention
    
    def forward(self, x, pos_embd_image=None, query_embed=None):
        if not self.use_pos_embd_image and not self.use_pos_embd_action: #everything learned - severe overfitting
            x = x + self.weight_pos_embed.weight[:, None]
        elif self.use_pos_embd_image and not self.use_pos_embd_action: #use learned positional embedding for action 
            x[-self.query_num:] = x[-self.query_num:] + self.weight_pos_embed.weight[:, None]
            x[:-self.query_num] = x[:-self.query_num] + pos_embd_image
        elif self.use_pos_embd_action and self.use_pos_embd_image: #all use sinsoidal positional embedding
            x[-self.query_num:] = x[-self.query_num:] + query_embed
            x[:-self.query_num] = x[:-self.query_num] + pos_embd_image 
                        
        x = self.attention_blocks(x)
        # take the last token
        return x