import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

# FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
FLASH_AVAILABLE = False     # naive attention

class MultiHeadAttention(nn.Module):
    def __init__(self, allow_flash, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.allow_flash = allow_flash
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # FlashAttention 또는 최적화된 Scaled Dot-Product Attention(SDP)을 활성화
        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Query, Key, Value 생성
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Query, Key, Value 분리

        if FLASH_AVAILABLE:
            if self.allow_flash:
                args = [x.half().contiguous() for x in [q, k, v]]
                attn_output = F.scaled_dot_product_attention(*args, attn_mask=None, dropout_p=self.dropout.p).to(q.dtype)     # 방법 1
                # attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p)                                                              # 방법 2
            else:
                args = [x.contiguous() for x in [q, k, v]]
                attn_output = F.scaled_dot_product_attention(*args, attn_mask=None, dropout_p=self.dropout.p)
        else:
            # Scaled Dot-Product Attention 계산
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)  # (batch_size, num_heads, seq_len, head_dim)
            

        # Multi-Head Attention 출력 변환
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        return output


class Transformer(nn.Module):
    def __init__(self,
                 allow_flash=True,
                 embed_dim=512,
                 num_heads=8,
                 ff_hidden_dim=2048,
                 seq_len=128,
                 num_classes=10,
                 dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(allow_flash,
                                            embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(embed_dim * seq_len // num_heads * num_heads,
                      num_classes)
        )

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        logits = self.classifier(x)

        return logits
