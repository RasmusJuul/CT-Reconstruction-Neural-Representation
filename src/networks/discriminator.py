import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import spectral_norm
import math
import numpy as np

#############################
# Self-Attention Module
#############################
class SelfAttention1d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        self.query_conv = spectral_norm(nn.Conv1d(in_dim, in_dim // 8, kernel_size=1))
        self.key_conv   = spectral_norm(nn.Conv1d(in_dim, in_dim // 8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv1d(in_dim, in_dim, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.size()
        proj_query = self.query_conv(x).view(B, -1, L)  
        proj_key   = self.key_conv(x).view(B, -1, L)    
        proj_value = self.value_conv(x).view(B, C, L)    
        attention = torch.bmm(proj_query.transpose(1, 2), proj_key)  
        attention = F.softmax(attention, dim=-1)                     
        out = torch.bmm(proj_value, attention.transpose(1, 2))     
        out = self.gamma * out + x
        return out

#####################################
# Extended Discriminator Implementation
#####################################
class Discriminator(nn.Module):
    def __init__(self, num_points, 
                 use_multiscale=False, 
                 use_dilated=False, 
                 use_fft_branch=False, 
                 use_segment_aggregation=False,
                 num_segments=4,
                 use_transformer=False,
                 transformer_nhead=4,
                 transformer_num_layers=1,):
        """
        Args:
            num_points (int): Number of sample points per ray.
            use_multiscale (bool): If True, process each ray at multiple resolutions.
            use_dilated (bool): If True, add an extra branch with parallel dilated convolutions.
            use_fft_branch (bool): If True, add an FFT-based feature branch.
            use_segment_aggregation (bool): If True, split the ray into segments and aggregate features.
            num_segments (int): Number of segments to use in the segment aggregation branch.
        """
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_dilated = use_dilated
        self.use_fft_branch = use_fft_branch
        self.use_segment_aggregation = use_segment_aggregation
        self.num_segments = num_segments
        self.num_points = num_points  # needed for FFT and segment branches


        

        ###############################
        # Base Convolutional Pipeline
        ###############################
        self.conv1 = spectral_norm(nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3))
        self.conv2 = spectral_norm(nn.Conv1d(16, 32, kernel_size=3, padding=1))
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.skip1 = spectral_norm(nn.Conv1d(16, 32, kernel_size=1))
        
        if self.use_dilated:
            # Use dilation in conv3 if requested.
            self.conv3 = spectral_norm(nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2))
        else:
            self.conv3 = spectral_norm(nn.Conv1d(32, 64, kernel_size=3, padding=1))
        self.conv4 = spectral_norm(nn.Conv1d(64, 32, kernel_size=3, padding=1))
        self.skip2 = spectral_norm(nn.Conv1d(64, 32, kernel_size=1))
        self.conv5 = spectral_norm(nn.Conv1d(32, 16, kernel_size=3, padding=1))
        self.conv6 = spectral_norm(nn.Conv1d(16, 8, kernel_size=3, padding=1))
        
        #####################################
        # Optional Dilated Convolution Branch
        #####################################
        if self.use_dilated:
            self.dilated_branch = nn.ModuleList([
                spectral_norm(nn.Conv1d(32, 32, kernel_size=3, padding=d, dilation=d))
                for d in [1, 2, 4]
            ])
            self.dilated_conv_out = nn.Conv1d(32 * 3, 8, kernel_size=1)
        
        ##########################
        # Self-Attention and Output
        ##########################
        self.attention = SelfAttention1d(8)
        self.conv7 = spectral_norm(nn.Conv1d(8, 1, kernel_size=3, padding=1))
        # The base branch (or multiscale branch) always produces one feature per ray.
        # We'll refer to that as the "base" feature.

        ###############################
        # Optional Transformer
        ###############################
        self.use_transformer = use_transformer
        if self.use_transformer:
            # Here the feature channels from the conv pipeline are 8 (from conv6)
            # Create a transformer encoder layer; note that the transformer expects (L, B, C)
            encoder_layer = TransformerEncoderLayer(d_model=8, nhead=transformer_nhead, dropout=0.1)
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        ##############################################
        # Optional FFT Branch: Feature from frequency domain
        ##############################################
        if self.use_fft_branch:
            # Compute FFT size from num_points; we use rfft so output length is floor(n/2)+1.
            self.fft_length = self.num_points // 2 + 1
            self.fft_mlp = nn.Sequential(
                nn.Linear(self.fft_length, 32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, 1)
            )
        
        ##################################################
        # Optional Segment Aggregation Branch
        ##################################################
        if self.use_segment_aggregation:
            # Compute approximate segment length (assumes num_points divisible by num_segments)
            self.segment_length = self.num_points // self.num_segments
            # A simple MLP to process each segment (flattened segment of ray values)
            self.segment_mlp = nn.Sequential(
                nn.Linear(self.segment_length, 32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, 1)
            )

        ##################################################
        # Optional Learned fusion of multiscale
        ##################################################
        if self.use_multiscale:
            self.scale_fusion = nn.Sequential(
                nn.Linear(3, 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(8, 1),
            )
        
        ##############################################
        # Final MLP for Classification
        ##############################################
        # Compute the total number of features that will be concatenated.
        # Always: base feature (1)
        final_feature_dim = 1
        if self.use_fft_branch:
            final_feature_dim += 1
        if self.use_segment_aggregation:
            final_feature_dim += 1
        
        # Plus 6 for start and end points (3 each)
        mlp_input_dim = final_feature_dim + 6
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    
    def process_ray(self, x):
        """
        Process a ray (tensor shape: (B, 1, L)) through the convolutional pipeline.
        Returns a tensor of shape (B, 1) representing the base feature.
        """
        x1 = self.leaky_relu(self.conv1(x))
        x1 = self.dropout(x1)
        x2 = self.leaky_relu(self.conv2(x1))
        x2 = self.dropout(x2)
        x2 = x2 + self.skip1(x1)
        x3 = self.leaky_relu(self.conv3(x2))
        x3 = self.dropout(x3)
        x4 = self.leaky_relu(self.conv4(x3))
        x4 = self.dropout(x4)
        x4 = x4 + self.skip2(x3)
        x5 = self.leaky_relu(self.conv5(x4))
        x5 = self.dropout(x5)
        x6 = self.leaky_relu(self.conv6(x5))
        x6 = self.dropout(x6)
        
        if self.use_dilated:
            dilated_features = []
            for conv in self.dilated_branch:
                dilated_features.append(conv(x2))
            dilated_features = torch.cat(dilated_features, dim=1)
            dilated_features = self.dilated_conv_out(dilated_features)
            x6 = x6 + dilated_features
        
        # Instead of immediately pooling over the spatial dimension, apply transformer:
        if self.use_transformer:
            # x6 is (B, C, L'). Permute to (L', B, C) as required by the transformer.
            x6_seq = x6.permute(2, 0, 1)  # now (L', B, C)
            x6_seq = self.transformer_encoder(x6_seq)
            # Permute back and then average pool over the sequence dimension.
            x6 = x6_seq.permute(1, 2, 0)  # (B, C, L')
        
        # Global average pooling along the spatial dimension to obtain one feature per ray
        x_final = self.conv7(self.attention(x6))  # shape: (B, 1, L')
        features = x_final.squeeze(1).mean(dim=1, keepdim=True)  # (B, 1)
        return features
    
    def extract_base_feature(self, ray):
        """
        Extract the base feature from the ray.
        If use_multiscale is enabled, process at multiple scales and fuse.
        Otherwise, process at full resolution.
        Input ray shape: (B, L)
        Output: tensor of shape (B, 1)
        """
        x = ray.unsqueeze(1)  # shape: (B, 1, L)
        if not self.use_multiscale:
            return self.process_ray(x)
        else:
            features_list = []
            # Full resolution
            feat_full = self.process_ray(x)
            features_list.append(feat_full)
            # Half resolution
            x_half = F.avg_pool1d(x, kernel_size=2, stride=2)
            feat_half = self.process_ray(x_half)
            features_list.append(feat_half)
            # Quarter resolution
            x_quarter = F.avg_pool1d(x, kernel_size=4, stride=4)
            feat_quarter = self.process_ray(x_quarter)
            features_list.append(feat_quarter)
            # Fuse features from each scale by concatenation and a learned linear layer.
            fused_feats = torch.cat(features_list, dim=1)
            # Learnable fusion
            fused = self.scale_fusion(fused_feats)
            return fused  # shape: (B, 1)
    
    def fft_feature(self, ray):
        """
        Compute FFT of the ray and use a small MLP to get a single feature.
        Input ray: (B, L)
        """
        # Compute the real FFT along the last dimension.
        # fft_result: shape (B, fft_length)
        fft_result = torch.fft.rfft(ray, n=self.num_points).abs()
        # Optionally, one could take log(1+abs()) for dynamic range compression.
        fft_result = torch.log1p(fft_result)
        # Pass through the FFT MLP.
        feat_fft = self.fft_mlp(fft_result)
        return feat_fft  # (B, 1)
    
    def segment_feature(self, ray):
        """
        Divide the ray into segments, process each segment and aggregate.
        Input ray: (B, L)
        """
        B, L = ray.size()
        seg_len = self.segment_length  # assumed L is divisible by num_segments
        segments = torch.split(ray, seg_len, dim=1)  # list of (B, seg_len) tensors
        seg_features = []
        for seg in segments:
            feat = self.segment_mlp(seg)  # (B, 1)
            seg_features.append(feat)
        # Aggregate segment features; here we average.
        seg_features = torch.stack(seg_features, dim=1)  # shape: (B, num_segments, 1)
        seg_features = seg_features.mean(dim=1)  # shape: (B, 1)
        return seg_features
    
    def forward(self, ray, start_point, end_point):
        """
        Args:
            ray (Tensor): (B, L) attenuation values along the ray.
            start_point (Tensor): (B, 3)
            end_point (Tensor): (B, 3)
        Returns:
            validity (Tensor): (B, 1)
        """
        # Gather features from the various branches.
        features = []
        
        # Base feature (from conv pipeline, possibly multiscale)
        base_feat = self.extract_base_feature(ray)  # (B, 1)
        features.append(base_feat)
        
        # FFT branch feature, if enabled.
        if self.use_fft_branch:
            fft_feat = self.fft_feature(ray)  # (B, 1)
            features.append(fft_feat)
        
        # Segment aggregation branch, if enabled.
        if self.use_segment_aggregation:
            seg_feat = self.segment_feature(ray)  # (B, 1)
            features.append(seg_feat)
        
        # Concatenate all features along dimension 1.
        combined_features = torch.cat(features, dim=1)  # shape: (B, final_feature_dim)
        # Append the start and end point information.
        mlp_input = torch.cat([combined_features, start_point, end_point], dim=1)
        validity = self.mlp(mlp_input)
        return validity