import torch
from torch import nn

class DynamicFusionMLP(nn.Module):
    def __init__(self, text_dim, vision_dim, embed_dim, hidden_dim, num_stages=1):
        super(DynamicFusionMLP, self).__init__()
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.num_stages = num_stages


        self.fusion_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_stages)
        ])


        self.dynamic_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)
            )
            for _ in range(num_stages)
        ])

    def forward(self, text_feat, vision_feat):

        text_proj = self.text_proj(text_feat)  # [batch_size, embed_dim]


        vision_proj = self.vision_proj(vision_feat)  # [batch_size, seq_length, embed_dim]


        text_proj = text_proj.unsqueeze(1).expand(-1, vision_proj.size(1), -1)


        fused_feat = vision_proj


        for i in range(self.num_stages):
            fusion_input = torch.cat([text_proj, fused_feat], dim=-1)
            weight = self.fusion_gate[i](fusion_input)
            fused_feat = weight * text_proj + (1 - weight) * fused_feat
            fused_feat = self.dynamic_mlp[i](fused_feat)

        return fused_feat
