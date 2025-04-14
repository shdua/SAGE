import numpy as np
import torch
from torch import nn
from torchcrf import CRF
import torchvision.models as models
from transformers import BertModel, XLMRobertaModel
import torch.nn.functional as F


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthWiseConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PAM_Module(nn.Module):
    """Position Attention Module with Text Feature Integration"""
    def __init__(self, in_dim, text_dim):
        super(PAM_Module, self).__init__()
        self.query_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.key_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.value_conv = DepthWiseConv2d(in_dim, in_dim, kernel_size=1)
        self.text_fc = nn.Linear(text_dim, in_dim)  # 文本特征投影
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, text_feat):
        m_batchsize, C, height, width = x.size()

        text_proj = self.text_fc(text_feat).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x + text_proj).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, negative_slope=0.1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):

        return self.act(self.bn(self.conv(torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))))


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, negative_slope=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DA_Block(nn.Module):
    def __init__(self, in_channels, text_dim):
        super(DA_Block, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv5c = nn.Sequential(DepthWiseConv2d(in_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.sa = PAM_Module(inter_channels, text_dim)
        self.conv51 = nn.Sequential(DepthWiseConv2d(inter_channels, inter_channels, 3, padding=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), DepthWiseConv2d(inter_channels, in_channels, 1), nn.ReLU())

    def forward(self, x, text_feat):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1, text_feat)
        sa_conv = self.conv51(sa_feat)
        sa_output1 = self.conv6(sa_conv)
        return sa_output1


class DynamicFusionMLP(nn.Module):
    def __init__(self, text_dim, vision_dim, embed_dim, hidden_dim):
        super(DynamicFusionMLP, self).__init__()
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.vision_projection = nn.Linear(vision_dim, embed_dim)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, text_feat, vision_feat):

        text_proj = self.text_projection(text_feat)
        vision_proj = self.vision_projection(vision_feat)
        

        concat_feat = torch.cat((text_proj.unsqueeze(1).expand(-1, vision_feat.size(1), -1), vision_proj), dim=2)
        fused_feat = self.fusion_mlp(concat_feat)
        
        return fused_feat


class ImageModel(nn.Module):
    def __init__(self, hidden_size, negative_slope1, negative_slope2, k=10, text_dim=None):
        super(ImageModel, self).__init__()
        self.hidden_size = hidden_size
        

        resnet = models.resnet50(pretrained=True)
        

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        

        self.adapter = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        

        self.da_block = DA_Block(256, text_dim)

    def forward(self, x, text_feat):

        x = self.backbone(x)
        

        x = self.adapter(x)
        

        enhanced_layer = self.da_block(x, text_feat)
        return enhanced_layer


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), dropout=dropout, attn_type=attn_type)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, attn_mask=attn_mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        if hasattr(self, 'fc'):
            output = self.fc(output)
        if hasattr(self, 'dropout'):
            output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.attn_type = nn.Softmax(dim=2) if attn_type == 'softmax' else nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e6)
        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class FANetModel(nn.Module):
    def __init__(self, label_list, args):
        super(FANetModel, self).__init__()
        self.args = args
        self.num_labels = len(label_list)
        

        self.text_encoder = XLMRobertaModel.from_pretrained(args.bert_name)
        self.bert_config = self.text_encoder.config
        

        self.vision_encoder = ImageModel(
            self.bert_config.hidden_size,
            args.negative_slope1,
            args.negative_slope2,
            args.dyn_k,
            text_dim=args.embed_dim,
        )
        

        self.dynamic_proj = DynamicFusionMLP(
            text_dim=self.bert_config.hidden_size,
            vision_dim=256,
            embed_dim=self.args.embed_dim,
            hidden_dim=128
        )
        
        self.Cross_atten = MultiHeadAttention(
            n_head=self.bert_config.num_attention_heads,
            d_model=self.bert_config.hidden_size,
            d_k=self.bert_config.hidden_size,
            d_v=self.bert_config.hidden_size,
        )
        
        self.classifier = nn.Linear(self.bert_config.hidden_size, self.num_labels)
        self.final_dropout = nn.Dropout(args.dropout_prob)
        self.crf = CRF(self.num_labels, batch_first=True)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, mode="train"):

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
  
        text_hidden_state, text_pooler_output = text_output.last_hidden_state, text_output.pooler_output
        

        vision_output = self.vision_encoder(images, text_pooler_output)
        vision_output_flat = vision_output.flatten(2).permute(0, 2, 1)
        

        fused_feat = self.dynamic_proj(text_pooler_output, vision_output_flat)
        

        cross_output, _ = self.Cross_atten(text_hidden_state, fused_feat, fused_feat)
        

        emissions = self.final_dropout(self.classifier(cross_output))
        logits = self.crf.decode(emissions, attention_mask.byte())
        

        loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction="mean") if labels is not None else None
        
        return logits, loss