import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import os
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from typing import Any
from torchvision import models
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



class GazeEncoder(nn.Module):
    """
    Gaze encoder for normalised x,y coordinates in frame space
    Input: (batch, num_frames, 2) - gaze x,y coordinates
    Output: (batch, num_frames, dim_feat)
    """
    def __init__(self, dim_feat, input_dim=2,  dropout=0.3, use_temporal=True):
        super(GazeEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.dim_feat = dim_feat
        self.use_temporal = use_temporal
        
        # Initial projection to higher dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dim_feat)
        )
        
        # 1D conv (temporal) layers
        self.temporal_conv1 = nn.Conv1d(dim_feat, dim_feat, kernel_size=3, padding=1)

    
    def forward(self, x):
        
        x_projected = self.input_projection(x) 
                
        # Apply temporal convs
        temporal_conv = self.temporal_conv1(x_projected.transpose(1,2)).transpose(1, 2)
        output = x_projected + temporal_conv  # Residual connection
        
        return output

class RGBEncoder(nn.Module):
    """
    RGB encoder for circular egocentric video frames
    Input: (batch, num_frames, channels, height, width)
    Output: (batch, num_frames, spatial_dim, dim_feat)
    """

    def __init__(self, dim_feat, input_dim=3, dropout=0.3, out_size=7):
        super(RGBEncoder, self).__init__()
        
        self.dim_feat = dim_feat
        self.out_size = out_size
        self.dropout = dropout
        
        # Load pretrained Swin Tiny for spatial features
        self.swin_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            num_classes=0,
            global_pool=''
        )


        # Motion backbone (ResNet18)
        self.motion_backbone = models.resnet18(pretrained=True)
        self.motion_backbone = nn.Sequential(*list(self.motion_backbone.children())[:-2]) # Remove final fully connected layer and avg pool for feature extraction
        
        # Modify first conv of backbone for motion via frame differences
        motion_conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.motion_backbone[0] = motion_conv1
        
        self.motion_proj = nn.Sequential(
            nn.Conv2d(512, dim_feat//2, kernel_size=1),
            nn.BatchNorm2d(dim_feat//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        # Temporal modeling 
        self.temporal_modeling = nn.Sequential(
            nn.Conv1d(dim_feat, dim_feat, kernel_size=3, padding=1, groups=max(1, dim_feat//8)),
            nn.BatchNorm1d(dim_feat),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_feat, dim_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_feat),
            nn.ReLU(inplace=True)
        )
        
        self.feature_proj = None
        self.motion_spatial_proj = None
        self.swin_output_dim = 768
        self.spatial_pool = nn.AdaptiveAvgPool2d((out_size, out_size))
        
    def forward(self, x):
        
        batch_size, num_frames = x.shape[:2]
        
        x_frames = x.view(batch_size * num_frames, *x.shape[2:])
        x_frames = x_frames.permute(0, 3,1,2) 

        swin_features = self.swin_backbone(x_frames)
        
        swin_pooled = self.spatial_pool(swin_features)  # (batch*frames, 768, out_size, out_size)
            
        bf, c, _, _ = swin_pooled.shape
        swin_features_flat = swin_pooled.view(bf, c, -1).permute(0, 2, 1)  # (bf, spatial_locs, channels)
        
        # create appearance projection layer dynamically 
        if self.feature_proj is None:
            input_dim = swin_features_flat.shape[-1]
            self.feature_proj = nn.Sequential(
                nn.Linear(input_dim, self.dim_feat//2),  # dim_feat//2 for concatenation
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ).to(swin_features_flat.device)
        
        
        appearance_features = self.feature_proj(swin_features_flat)  
        
        x_motion = x[:, 1:] - x[:, :-1]  # frame differences
        x_motion = F.pad(x_motion, (0, 0, 0, 0, 0, 0, 1, 0)) # padding to keep same number of frames
        
        # reshaping motion frames
        x_motion_frames = x_motion.view(batch_size * num_frames, *x_motion.shape[2:])
        x_motion_frames = x_motion_frames.permute(0, 3, 1, 2)  

        # motion feature extraction
        motion_backbone_out = self.motion_backbone(x_motion_frames) 
        motion_features = self.motion_proj(motion_backbone_out) 
        
        # Pool motion features to match appearance spatial dimensions
        motion_pooled = self.spatial_pool(motion_features) 
        
        bf, c, _, _ = motion_pooled.shape
        motion_features_flat = motion_pooled.view(bf, c, -1).permute(0, 2, 1) 
        
        # fuse spatial and motion features
        fused_features = torch.cat([appearance_features, motion_features_flat], dim=-1) 
        
        # reshape to video format
        spatial_locations = fused_features.shape[1]
        video_features = fused_features.view(batch_size, num_frames, spatial_locations, self.dim_feat)
        
        # temporal modeling on video features
        b, f, s, d = video_features.shape
        temporal_input = video_features.permute(0, 2, 3, 1).reshape(b * s, d, f)  
        temporal_output = self.temporal_modeling(temporal_input)
        
        final_features = temporal_output.view(b, s, d, f).permute(0, 3, 1, 2)  # (batch, frames, spatial, dim_feat)
        
        return final_features

class IMUEncoder(nn.Module):
    """
    IMU encoder for motion features
    Input: (batch, num_frames, seq_len, input_dim)
    Output: (batch, num_frames, dim_feat)
    """
    def __init__(self, input_dim=6, dim_feat=256, dropout=0.3):
        super(IMUEncoder, self).__init__()
        
        # individual frame IMU encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, dim_feat, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # GRU for temporal features
        self.temporal_gru = nn.GRU(
            input_size=dim_feat,
            hidden_size=dim_feat,
            num_layers=1,
            batch_first=True
        )

        # layer norm for stability
        self.norm = nn.LayerNorm(dim_feat)
        
    def forward(self, x):

        _, num_frames, _, _ = x.shape
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame_data = x[:, i, :, :] # imu data for frame i
            frame_data = frame_data.permute(0, 2, 1)
            
            # Encode frame
            encoded = self.frame_encoder(frame_data).squeeze(-1)  # (batch, dim_feat)
            frame_features.append(encoded)
        
        # stack temporal features
        temporal_features = torch.stack(frame_features, dim=1)  # (batch, num_frames, dim_feat)
        
        # Apply temporal GRU
        gru_out, _ = self.temporal_gru(temporal_features)
        
        # Normalize
        output = self.norm(gru_out)
        
        return output

class ObjectDetectionEncoder(nn.Module):
    """
    Encodes object detection feature vectors 

    Input: (batch, 32, num_objects * features_per_object)
    Output: (batch, 32, dim_feat)
    """

    def __init__(self, dim_feat, num_objects=4, features_per_object=5, dropout=0.3):
        super(ObjectDetectionEncoder, self).__init__()

        self.num_objects = num_objects
        self.features_per_object = features_per_object


        # simple spatial feature encoder
        self.simple_encoder = nn.Sequential(
            nn.Linear(num_objects * features_per_object, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, dim_feat)
        )

        # temporal convolution 
        self.temporal_conv = nn.Conv1d(dim_feat, dim_feat, kernel_size=3, padding=1)
       

    def forward(self, x):
        
        #spatial features
        simple_encoded = self.simple_encoder(x)  
        
        #temporal features
        temporal_conv = self.temporal_conv(simple_encoded.transpose(1,2)).transpose(1, 2)  # (batch, num_frames, dim_feat)

        # Residual connection & non-learning based fusion 
        output = simple_encoded + temporal_conv  

        return output

class HandEncoder(nn.Module):

    """
    Hand encoder that learns from both present landmarks and their absence patterns
    Input: (batch, num_frames, input_dim) - may contain NaN values
    Output: (batch, num_frames, dim_feat)
    """
    def __init__(self, dim_feat, input_dim=8, dropout=0.3, num_frames=32):
        super(HandEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.dim_feat = dim_feat
        self.num_frames = num_frames
        
        # learnable missing value embeddings
        self.missing_token = nn.Parameter(torch.randn(input_dim) * 0.02)
        
        
        # spatial feature encoder for landmark values
        self.value_encoder = nn.Sequential(
            nn.Linear(input_dim, dim_feat // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat // 2, dim_feat // 2)
        )
        
        # validity encoder (for presence/absence patterns)
        self.validity_encoder = nn.Sequential(
            nn.Linear(input_dim, dim_feat // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat // 4, dim_feat // 4)
        )
        
        # missing pattern encoder (for temporal absence patterns)
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, dim_feat // 4),  # Frame + landmark missing ratios
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat // 4, dim_feat // 4)
        )
        
        # fusion layer to combine
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim_feat, dim_feat),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat, dim_feat)
        )
        
        # final temporal modeling with MHA 
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=dim_feat, num_heads=8, batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(dim_feat)
        
    def create_validity_mask(self, x):
        """Create binary mask indicating which values are valid (not NaN)"""
        return (~torch.isnan(x)).float()  # 1 where valid, 0 where NaN
    
    def compute_missing_patterns(self, validity_mask):
        """
        Compute meaningful features from missing value patterns
        """
        _, num_frames, input_dim = validity_mask.shape
        
        # missing ratio per frame
        frame_missing_ratio = 1.0 - validity_mask.mean(dim=2) 
        
        # missing ratio per landmark across time
        landmark_missing_ratio = 1.0 - validity_mask.mean(dim=1)
        
        # expand ratios to match landmark dimensions for concatenation
        frame_ratios_expanded = frame_missing_ratio.unsqueeze(2).expand(-1, -1, input_dim)
        landmark_ratios_expanded = landmark_missing_ratio.unsqueeze(1).expand(-1, num_frames, -1)
        
        combined_patterns = torch.cat([
            frame_ratios_expanded,     
            landmark_ratios_expanded    
        ], dim=2) 
        
        return combined_patterns
    
    def handle_nan_values(self, x):
        """
        Process NaN values in hand landmarks
        """
        validity_mask = self.create_validity_mask(x)
        
        x_filled = torch.where(torch.isnan(x), 
                              self.missing_token.expand_as(x), 
                              x)
        
        missing_patterns = self.compute_missing_patterns(validity_mask)
        
        return x_filled, validity_mask, missing_patterns
    
    def forward(self, x):
        """
        Forward pass that handles NaN values and learns from absence patterns
        
        Args:
            x: (batch, frames, input_dim) - may contain NaN values
            
        Returns:
            features: (batch, frames, dim_feat)
        """
        
        # NaN handling and pattern extraction
        x_filled, validity_mask, missing_patterns = self.handle_nan_values(x)
        
        # spatial
        value_features = self.value_encoder(x_filled) 
        
        # validity
        validity_features = self.validity_encoder(validity_mask)  
        
        # missing patterns
        pattern_features = self.pattern_encoder(missing_patterns) 
        
        # combine all features
        combined_features = torch.cat([
            value_features,    
            validity_features,  
            pattern_features    
        ], dim=2) 
        

        # fusion
        fused_features = self.fusion_layer(combined_features)
        
        # temporal 
        attended_features, _ = self.temporal_attention(
            fused_features, fused_features, fused_features
        )
        

        # residual connection and normalization
        output_features = self.layer_norm(attended_features + fused_features)
        
        return output_features  
    
class TemporalPooling(nn.Module):
    """Pools temporal dimension for video sequences"""
    def __init__(self, dim):
        super().__init__()
        
        self.pool_token = nn.Parameter(torch.randn(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)


    def forward(self, x):
        
        b, f, n, d = x.shape
        
        pool_token = self.pool_token.expand(b, f, -1, -1)
        x_with_pool = torch.cat([pool_token, x], dim=2)

        # Reshape for attention
        x_flat = x_with_pool.view(b * f, n + 1, d)
        pooled, _ = self.attention(x_flat[:, :1], x_flat, x_flat)
        pooled = pooled.view(b, f, d)

        return pooled

class TemporalTransformerBlock(nn.Module):
    """Transformer block with temporal positional encoding"""
    def __init__(self, dim, heads=8, dim_head=64, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):

        # Self-attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x

class EgoDriveMultimodalTransformer(nn.Module):
    def __init__(
        self,
        dim_feat,
        num_classes=6,
        dropout=0.3,
        num_frames=32,
        transformer_depth=2,
        transformer_heads=4
    ):
        super().__init__()

        # unimodal encoders
        self.rgb_encoder = RGBEncoder( dim_feat=dim_feat, dropout=dropout)
        self.object_encoder = ObjectDetectionEncoder( num_objects = 4, features_per_object=5, dim_feat=dim_feat, dropout=dropout)
        self.gaze_encoder = GazeEncoder(input_dim=2, dim_feat=dim_feat, dropout=dropout)
        self.hand_encoder = HandEncoder(dim_feat=dim_feat, input_dim=8, dropout=dropout, num_frames=num_frames)
        self.imu_encoder = IMUEncoder(input_dim=6, dim_feat=dim_feat, dropout=dropout)

        # temporal pooling
        self.temporal_pool = TemporalPooling(dim_feat)

        # learnable modality embeddings
        self.modality_embeddings = nn.Parameter(torch.randn(5, dim_feat))

        # temporal positional encoding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, dim_feat))


        # transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TemporalTransformerBlock(dim_feat, transformer_heads, dropout=dropout)
            for _ in range(transformer_depth)
        ])

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_feat),
            nn.Linear(dim_feat, dim_feat),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feat, num_classes)
        )

        # Spatial attention for RGB frames
        self.spatial_attention = nn.MultiheadAttention(
        embed_dim=dim_feat, num_heads=8, batch_first=True
        )
    
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_feat) * 0.02)

    def forward(self, inputs, labels=None):
        batch_size = list(inputs.values())[0].shape[0]
        encoded_modalities = []
        
        
        if 'frames' in inputs:
            # Encode RGB frames
            rgb_features = self.rgb_encoder(inputs['frames'])  

            batch_size, num_frames, num_spatial, dim = rgb_features.shape
            rgb_reshaped = rgb_features.contiguous().view(batch_size * num_frames, num_spatial, dim)

            # Apply spatial attention
            rgb_attended, _ = self.spatial_attention(rgb_reshaped, rgb_reshaped, rgb_reshaped)
            
            # reshape for concatenation
            rgb_pooled = rgb_attended.mean(dim=1) 
            rgb_features = rgb_pooled.view(batch_size, num_frames, dim)
            
            # add modality embedding
            rgb_features = rgb_features + self.modality_embeddings[0]
            encoded_modalities.append(rgb_features)
        

        # Encode other modalities (all produce (batch, 32, dim) outputs)
        if 'objects' in inputs:
            obj_features = self.object_encoder(inputs['objects'])
            obj_features = obj_features + self.modality_embeddings[1]
            encoded_modalities.append(obj_features)
        
        if 'gaze' in inputs:
            gaze_features = self.gaze_encoder(inputs['gaze'])
            gaze_features = gaze_features + self.modality_embeddings[2]
            encoded_modalities.append(gaze_features)
        
        if 'hands' in inputs:
            hand_features = self.hand_encoder(inputs['hands'])
            hand_features = hand_features + self.modality_embeddings[3]
            encoded_modalities.append(hand_features)
        
        if 'imu' in inputs:
            imu_features = self.imu_encoder(inputs['imu'])
            imu_features = imu_features + self.modality_embeddings[4]
            encoded_modalities.append(imu_features)
        
        # handles unimodal testing
        if len(encoded_modalities) > 1:

            min_frames = min(feat.shape[1] for feat in encoded_modalities)
            encoded_modalities = [feat[:, :min_frames] for feat in encoded_modalities]
            stacked = torch.stack(encoded_modalities, dim=0)  
            fused_features = stacked.mean(dim=0)
        else:
            fused_features = encoded_modalities[0]
        
        
        # Store initial features for residual connections
        initial_features = fused_features.clone()
        
        seq_len = fused_features.shape[1]
        
        if seq_len > self.temporal_pos_embed.shape[1]:
            # Interpolate if sequence is longer than expected
            temp_pos = F.interpolate(
                self.temporal_pos_embed.permute(0, 2, 1), 
                size=seq_len, 
                mode='linear'
            ).permute(0, 2, 1)
            fused_features = fused_features + temp_pos
        else:
            fused_features = fused_features + self.temporal_pos_embed[:, :seq_len, :]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate CLS token with modality features
        features = torch.cat([cls_tokens, fused_features], dim=1)  
        
        # apply temporal transformer blocks with residual connections
        for i, block in enumerate(self.transformer_blocks):
            features = block(features)
            
            # residual connection every second block
            if i % 2 == 1 and i > 0:
                features[:, 1:] = features[:, 1:] + initial_features * 0.1
        

        # extract CLS token for classification
        cls_output = features[:, 0] 
        
        # normalize features 
        cls_output = F.layer_norm(cls_output, (cls_output.shape[-1],))
        
        
        # Classification
        logits = self.classifier(cls_output)
        
        outputs = {
            'logits': logits,
            'features': cls_output,
            'temporal_features': features[:, 1:]  # Exclude CLS token
        }
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs['loss'] = loss
            
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == labels).float().mean()
                outputs['accuracy'] = accuracy
        
        return outputs