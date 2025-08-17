class PointwiseAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(PointwiseAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, nhead)
        )

    def forward(self, x, pos):
        batch_size, num_points, _ = x.size()
        q = self.q_proj(x).view(batch_size, num_points, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_points, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_points, self.nhead, self.head_dim)
        
        q = q.permute(2, 0, 1, 3).contiguous()  # [nhead, batch_size, num_points, head_dim]
        k = k.permute(2, 0, 1, 3).contiguous()
        v = v.permute(2, 0, 1, 3).contiguous()
        
        # Compute pairwise relative positions
        pos_pairwise = pos.unsqueeze(1) - pos.unsqueeze(2)  # [batch_size, num_points, num_points, 3]
        pos_pairwise_flat = pos_pairwise.view(batch_size, num_points * num_points, 3)
        pos_encoding = self.pos_mlp(pos_pairwise_flat)  # [batch_size, num_points*num_points, nhead]
        pos_encoding = pos_encoding.view(batch_size, num_points, num_points, self.nhead)  # [batch_size, num_points, num_points, nhead]
        
        # Compute attention logits
        attn_logits = torch.einsum('hbid,hbjd->hbij', q, k) * self.scaling  # [nhead, batch_size, num_points, num_points]
        attn_logits = attn_logits.permute(1, 2, 3, 0).contiguous()  # [batch_size, num_points, num_points, nhead]
        attn_logits += pos_encoding  # Add position-based attention
        attn_logits = attn_logits.permute(3, 0, 1, 2).contiguous()  # [nhead, batch_size, num_points, num_points]
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.einsum('hbij,hbjd->hbid', attn_weights, v)
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(batch_size, num_points, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):#dim_feedforward=2048
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = PointwiseAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, pos):
        src2 = self.self_attn(src, pos)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class PointTransformerClassification(nn.Module):
    def __init__(self, num_classes, d_model=64, nhead=4, num_layers=3):
        super(PointTransformerClassification, self).__init__()
        self.embedding = nn.Linear(3, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        xyz = x  # Original coordinates
        x = self.embedding(x)  # Features
        for layer in self.encoder:
            x = layer(x, xyz)
        x = x.max(dim=1)[0]  # Global max pooling
        x = self.fc(x)
        return x
