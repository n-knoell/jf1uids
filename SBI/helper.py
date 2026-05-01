import torch
import pickle
from sbi.inference import NPE
from sbi.utils import BoxUniform
import torch.nn as nn
from torch.distributions import Distribution



## OLD for single frame:
class DensityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, 64, 64)
        x = x.unsqueeze(1)   # -> (B, 1, 64, 64)
        return self.conv(x)


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    elif name == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()

#OLD for single frame (tunable):
class DensityEmbeddingComplex(nn.Module):
    def __init__(self,
                 input_shape=(64,64),
                 num_conv_blocks=3,
                 base_channels=16,
                 kernel_size=3,
                 use_batchnorm=True,
                 dropout=0.0,
                 fc_dim=128,
                 activation='relu'):
        """
        input_shape: (H, W) of single-channel image
        num_conv_blocks: number of conv blocks (each downsamples by stride=2)
        base_channels: channels in first block; typically doubles each block
        kernel_size: 3 or 5
        use_batchnorm: add batchnorm after conv
        dropout: dropout prob before final fc
        fc_dim: final embedding dim
        activation: 'relu' or 'leaky_relu' or 'gelu'
        """
        super().__init__()
        assert input_shape[0] == input_shape[1], "code assumes square images; adapt if not."
        self.activation = get_activation(activation)
        ks = kernel_size
        padding = ks // 2

        convs = []
        in_ch = 1
        ch = base_channels
        for b in range(num_conv_blocks):
            convs.append(nn.Conv2d(in_ch, ch, kernel_size=ks, stride=2, padding=padding))
            if use_batchnorm:
                convs.append(nn.BatchNorm2d(ch))
            convs.append(self.activation)
            in_ch = ch
            ch = min(ch * 2, 512)  # avoid runaway channels
        self.conv = nn.Sequential(*convs)

        # compute flattened feature size after downsampling
        H = input_shape[0] // (2**num_conv_blocks)
        W = input_shape[1] // (2**num_conv_blocks)
        flattened = in_ch * H * W
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(flattened, fc_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
        # final embedding has dimension fc_dim
    def forward(self, x):
        # expect x shape (B, H, W) or (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.head(x)
        return x  # shape (B, fc_dim)


## NEW for temporal data (3D conv):
class DensityEmbeddingComplexTemporal(nn.Module):
    """
    Spatio-temporal embedding for inputs shaped (T, H, W) per-sample,
    """
    def __init__(
        self,
        input_shape=(10, 64, 64),     # (T, H, W)
        num_conv_blocks=3,
        base_channels=16,
        kernel_size=3,
        use_batchnorm=True,
        dropout=0.0,
        fc_dim=128,
        activation='relu',
        temporal_ks=None,         # 
        temporal_stride=1,       # 
    ):
        super().__init__()
        assert len(input_shape) == 3, "input_shape should be (T, H, W)"
        T, H, W = input_shape
        assert H == W, "this implementation assumes square spatial dims (H == W)"
        self.activation = get_activation(activation)

        ks = kernel_size
        # temporal kernel: preserve old behavior if temporal_ks not set
        if temporal_ks is None:
            temporal_ks = 3 if T >= 3 else 1
        pad_temporal = temporal_ks // 2
        pad_spatial = ks // 2

        convs = []
        in_ch = 1  # we treat data as single channel in the channel dimension for Conv3d
        ch = base_channels

        # Build several Conv3D blocks. Each block downsamples spatial dims by stride 2,
        # temporal stride is controlled by temporal_stride (default 1 to preserve T).
        for _ in range(num_conv_blocks):
            convs.append(
                nn.Conv3d(
                    in_ch,
                    ch,
                    kernel_size=(temporal_ks, ks, ks),
                    stride=(temporal_stride, 2, 2),   # temporal_stride injected here
                    padding=(pad_temporal, pad_spatial, pad_spatial),
                )
            )
            if use_batchnorm:
                convs.append(nn.BatchNorm3d(ch))
            convs.append(self.activation)
            in_ch = ch
            ch = min(ch * 2, 512)

        self.conv = nn.Sequential(*convs)

        # compute flattened size after convs:
        def conv_output_length(L_in, kernel, stride, padding, dilation=1):
            return (L_in + 2*padding - dilation*(kernel - 1) - 1) // stride + 1

        T_out = T
        H_out = H
        W_out = W
        for _ in range(num_conv_blocks):
            T_out = conv_output_length(T_out, temporal_ks, temporal_stride, pad_temporal)
            H_out = conv_output_length(H_out, ks, 2, pad_spatial)
            W_out = conv_output_length(W_out, ks, 2, pad_spatial)

        if H_out < 1 or W_out < 1 or T_out < 1:
            raise ValueError("num_conv_blocks or temporal settings too large for given input dims")

        flattened = in_ch * T_out * H_out * W_out
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(flattened, fc_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        Accepts:
          - x shape (B, T, H, W)  OR
          - x shape (B, 1, T, H, W) OR
          - x shape (B, C, T, H, W) (will be mean-pooled over channel dim -> (B,1,T,H,W))

        Returns:
          - embedding: (B, fc_dim)
        """
        if x.dim() == 4:
            # (B, T, H, W) -> (B, 1, T, H, W)
            x = x.unsqueeze(1)
        elif x.dim() == 5 and x.shape[1] != 1:
            # (B, C, T, H, W) with C != 1 -> collapse channel by mean
            x = x.mean(dim=1, keepdim=True)

        # now x is (B, 1, T, H, W)
        x = self.conv(x)          # -> (B, C', T', H', W')
        x = self.flatten(x)       # -> (B, flattened)
        x = self.head(x)          # -> (B, fc_dim)
        return x

# NEW for temporal data 2D + 1D (temporal) (tunable):
class DensityEmbeddingPerFrameTemporalAdv(nn.Module):
    """
    Per-frame 2D encoder -> per-frame embedding E -> temporal aggregator ->
    final fc output of size fc_dim.

    """
    def __init__(
        self,
        input_shape=(10, 64, 64),   # (T, H, W)
        num_conv_blocks=3,
        base_channels=16,
        kernel_size=3,
        use_batchnorm=False,
        dropout=0.0,
        fc_dim=128,
        activation='relu',
        temporal_pool_output=1,      
        per_frame_dim=None,         
        temporal_num_layers=2,
        temporal_stride_first=1,
    ):
        super().__init__()
        assert len(input_shape) == 3, "input_shape should be (T, H, W)"
        T, H, W = input_shape
        assert H == W, "this implementation assumes square spatial dims (H == W)"

        try:
            self.activation = get_activation(activation)
        except NameError:
            # fallback
            self.activation = nn.ReLU()

        ks = kernel_size
        padding = ks // 2

        # --- build 2D encoder (shared across frames) ---
        convs = []
        in_ch = 1
        ch = base_channels
        for _ in range(num_conv_blocks):
            convs.append(nn.Conv2d(in_ch, ch, kernel_size=ks, stride=2, padding=padding))
            if use_batchnorm:
                convs.append(nn.BatchNorm2d(ch))
            convs.append(self.activation)
            in_ch = ch
            ch = min(ch * 2, 512)
        self.frame_encoder = nn.Sequential(*convs)
        self.frame_encoder_out_channels = in_ch  # channels after last block

        # spatial pooling to compress each frame to a vector of length frame_encoder_out_channels
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # temporal pool output (how many temporal bins to keep; 1 means global avg)
        self.temporal_pool_output = int(temporal_pool_output)

        # optionally reduce per-frame vector to a smaller E_dim
        if per_frame_dim is None:
            per_frame_dim = self.frame_encoder_out_channels
        self.per_frame_dim = per_frame_dim
        if per_frame_dim != self.frame_encoder_out_channels:
            self.per_frame_proj = nn.Linear(self.frame_encoder_out_channels, per_frame_dim)
        else:
            self.per_frame_proj = None

        # --- temporal aggregator
        layers = []
        for i in range(int(temporal_num_layers)):
            stride = int(temporal_stride_first) if i == 0 else 1
            layers.append(nn.Conv1d(per_frame_dim, per_frame_dim, kernel_size=3, padding=1, stride=stride))
            layers.append(self.activation)
        self.temporal_conv = nn.Sequential(*layers)

        # final temporal pooling: convert temporal axis -> temporal_pool_output bins
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.temporal_pool_output)

        # final head input size depends on temporal_pool_output
        head_in = per_frame_dim * self.temporal_pool_output
        self.head = nn.Sequential(
            nn.Linear(head_in, fc_dim),
            self.activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x):
        """
        x can be (B, T, H, W), (B, 1, T, H, W), or (B, C, T, H, W).
        returns: (B, fc_dim)
        """
        if x.dim() == 5 and x.shape[1] != 1:
            # (B, C, T, H, W) -> mean over channel dim
            x = x.mean(dim=1)     # -> (B, T, H, W)
        elif x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)      # -> (B, T, H, W)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected input dims {x.shape}; expected (B, T, H, W) or (B, C, T, H, W)")

        B, T, H, W = x.shape
        x_frames = x.reshape(B * T, 1, H, W)

        # encode frames
        feat = self.frame_encoder(x_frames)       
        feat = self.spatial_pool(feat)           
        feat = feat.view(B * T, self.frame_encoder_out_channels)  # -> (B*T, C_enc)

        # optionally project to per-frame embedding dim
        if self.per_frame_proj is not None:
            feat = self.per_frame_proj(feat)       # -> (B*T, per_frame_dim)

        # reshape to (B, T, per_frame_dim)
        feat = feat.view(B, T, self.per_frame_dim)

        feat = feat.permute(0, 2, 1).contiguous()  # -> (B, per_frame_dim, T)
        feat = self.temporal_conv(feat)            # -> (B, per_frame_dim, T_out)

        # pool across time -> (B, per_frame_dim, P)
        feat = self.temporal_pool(feat)            
        feat = feat.view(B, -1)                   

        # final head -> (B, fc_dim)
        out = self.head(feat)
        return out

def make_embedding_from_trial(trial, input_shape=(64,64)):
    num_conv_blocks = trial.suggest_categorical("num_conv_blocks", [2, 3])                      
    # base_channels    = trial.suggest_categorical("base_channels", [16, 32])         
    # dropout          = trial.suggest_categorical("dropout", [0.0, 0.12, 0.3])           # light regularization
    fc_dim           = trial.suggest_categorical("fc_dim", [64, 128])           # embedding size choices
    # num_conv_blocks = 2
    base_channels = 16
    dropout = 0.0
    # fc_dim = 64
    # keep these fixed for stability / less branching
    kernel_size = 3
    use_batchnorm = False
    activation = 'relu'

    embedding = DensityEmbeddingComplex(
        input_shape=input_shape,
        num_conv_blocks=num_conv_blocks,
        # base_channels=base_channels,
        kernel_size=kernel_size,
        use_batchnorm=use_batchnorm,
        dropout=dropout,
        fc_dim=fc_dim,
        activation=activation
    )

    return embedding


# def make_embedding_from_trial_temporal(trial, input_shape=(10,64,64)):
#     num_conv_blocks = trial.suggest_categorical("num_conv_blocks", [2, 3])
#     fc_dim           = trial.suggest_categorical("fc_dim", [64, 128, 256])   
#     base_channels    = trial.suggest_categorical("base_channels", [16, 32])          # small channel choices
#     dropout          = trial.suggest_categorical("dropout", [0.0, 0.12])           # light regularization

#     # New minimal temporal hyperparameters sampled from trial:
#     temporal_ks = trial.suggest_categorical("temporal_ks", [1, 3])      # 1 = no temporal context; 3 = local temporal context
#     temporal_stride = trial.suggest_categorical("temporal_stride", [1, 2])  # 1 = preserve time, 2 = downsample time

#     # Keep some fixed choices for stability, as you had before
#     # base_channels = 32
#     dropout = 0.0
#     kernel_size = 3
#     use_batchnorm = False
#     activation = 'relu'

#     embedding = DensityEmbeddingComplexTemporal(
#         input_shape=input_shape,
#         num_conv_blocks=num_conv_blocks,
#         base_channels=base_channels,
#         kernel_size=kernel_size,
#         use_batchnorm=use_batchnorm,
#         dropout=dropout,
#         fc_dim=fc_dim,
#         activation=activation,
#         temporal_ks=temporal_ks,           # <-- pass through
#         temporal_stride=temporal_stride,   # <-- pass through
#     )
#     return embedding
