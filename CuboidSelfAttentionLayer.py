import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import  _generalize_padding, _generalize_unpadding
from .utils import get_norm_layer, apply_initialization
from functools import lru_cache

def masked_softmax(att_score, mask, axis: int = -1):
    """Ignore the masked elements when calculating the softmax.
     The mask can be broadcastable.

    Parameters
    ----------
    att_score
        Shape (..., length, ...)
    mask
        Shape (..., length, ...)
        1 --> The element is not masked
        0 --> The element is masked
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]

    Returns
    -------
    att_weights
        Shape (..., length, ...)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        if att_score.dtype == torch.float16:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1E4)
        else:
            att_score = att_score.masked_fill(torch.logical_not(mask), -1E18)
        att_weights = torch.softmax(att_score, dim=axis) * mask
    else:
        att_weights = torch.softmax(att_score, dim=axis)
    return att_weights



@lru_cache()
def compute_cuboid_self_attention_mask(data_shape, cuboid_size, shift_size, strategy, padding_type, device):
    """Compute the shift window attention mask

    Parameters
    ----------
    data_shape
        Should be T, H, W
    cuboid_size
        Size of the cuboid
    shift_size
        The shift size
    strategy
        The decomposition strategy
    padding_type
        Type of the padding
    device
        The device

    Returns
    -------
    attn_mask
        Mask with shape (num_cuboid, cuboid_vol, cuboid_vol)
        The padded values will always be masked. The other masks will ensure that the shifted windows
        will only attend to those in the shifted windows.
    """
    T, H, W = data_shape
    pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0]
    pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1]
    pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2]
    data_mask = None
    # Prepare data mask
    if pad_t > 0  or pad_h > 0 or pad_w > 0:
        if padding_type == 'ignore':
            data_mask = torch.ones((1, T, H, W, 1), dtype=torch.bool, device=device)
            data_mask = F.pad(data_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    else:
        data_mask = torch.ones((1, T + pad_t, H + pad_h, W + pad_w, 1), dtype=torch.bool, device=device)
    if any(i > 0 for i in shift_size):
        if padding_type == 'ignore':
            data_mask = torch.roll(data_mask,
                                   shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    if padding_type == 'ignore':
        # (1, num_cuboids, cuboid_volume, 1)
        data_mask = cuboid_reorder(data_mask, cuboid_size, strategy=strategy)
        data_mask = data_mask.squeeze(-1).squeeze(0)  # (num_cuboid, cuboid_volume)
    # Prepare mask based on index
    shift_mask = torch.zeros((1, T + pad_t, H + pad_h, W + pad_w, 1), device=device)  # 1 T H W 1
    cnt = 0
    for t in slice(-cuboid_size[0]), slice(-cuboid_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-cuboid_size[1]), slice(-cuboid_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-cuboid_size[2]), slice(-cuboid_size[2], -shift_size[2]), slice(-shift_size[2], None):
                shift_mask[:, t, h, w, :] = cnt
                cnt += 1
    shift_mask = cuboid_reorder(shift_mask, cuboid_size, strategy=strategy)
    shift_mask = shift_mask.squeeze(-1).squeeze(0)  # num_cuboids, cuboid_volume
    attn_mask = (shift_mask.unsqueeze(1) - shift_mask.unsqueeze(2)) == 0  # num_cuboids, cuboid_volume, cuboid_volume
    if padding_type == 'ignore':
        attn_mask = data_mask.unsqueeze(1) * data_mask.unsqueeze(2) * attn_mask
    return attn_mask



def update_cuboid_size_shift_size(data_shape, cuboid_size, shift_size, strategy):
    """Update the

    Parameters
    ----------
    data_shape
        The shape of the data
    cuboid_size
        Size of the cuboid
    shift_size
        Size of the shift
    strategy
        The strategy of attention

    Returns
    -------
    new_cuboid_size
        Size of the cuboid
    new_shift_size
        Size of the shift
    """
    new_cuboid_size = list(cuboid_size)
    new_shift_size = list(shift_size)
    for i in range(len(data_shape)):
        if strategy[i] == 'd':
            new_shift_size[i] = 0
        if data_shape[i] <= cuboid_size[i]:
            new_cuboid_size[i] = data_shape[i]
            new_shift_size[i] = 0
    return tuple(new_cuboid_size), tuple(new_shift_size)



def cuboid_reorder(data, cuboid_size, strategy):
    """Reorder the tensor into (B, num_cuboids, bT * bH * bW, C)

    We assume that the tensor shapes are divisible to the cuboid sizes.

    Parameters
    ----------
    data
        The input data
    cuboid_size
        The size of the cuboid
    strategy
        The cuboid strategy

    Returns
    -------
    reordered_data
        Shape will be (B, num_cuboids, bT * bH * bW, C)
        num_cuboids = T / bT * H / bH * W / bW
    """
    B, T, H, W, C = data.shape
    num_cuboids = T // cuboid_size[0] * H // cuboid_size[1] * W // cuboid_size[2]
    cuboid_volume = cuboid_size[0] * cuboid_size[1] * cuboid_size[2]
    intermediate_shape = []

    nblock_axis = []
    block_axis = []
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == 'l':
            intermediate_shape.extend([total_size // block_size, block_size])
            nblock_axis.append(2 * i + 1)
            block_axis.append(2 * i + 2)
        elif ele_strategy == 'd':
            intermediate_shape.extend([block_size, total_size // block_size])
            nblock_axis.append(2 * i + 2)
            block_axis.append(2 * i + 1)
        else:
            raise NotImplementedError
    data = data.reshape((B,) + tuple(intermediate_shape) + (C, ))
    reordered_data = data.permute((0,) + tuple(nblock_axis) + tuple(block_axis) + (7,))
    reordered_data = reordered_data.reshape((B, num_cuboids, cuboid_volume, C))
    return reordered_data


def cuboid_reorder_reverse(data, cuboid_size, strategy, orig_data_shape):
    """Reverse the reordered cuboid back to the original space

    Parameters
    ----------
    data
    cuboid_size
    strategy
    orig_data_shape

    Returns
    -------
    data
        The recovered data
    """
    B, num_cuboids, cuboid_volume, C = data.shape
    T, H, W = orig_data_shape

    permutation_axis = [0]
    for i, (block_size, total_size, ele_strategy) in enumerate(zip(cuboid_size, (T, H, W), strategy)):
        if ele_strategy == 'l':
            # intermediate_shape.extend([total_size // block_size, block_size])
            permutation_axis.append(i + 1)
            permutation_axis.append(i + 4)
        elif ele_strategy == 'd':
            # intermediate_shape.extend([block_size, total_size // block_size])
            permutation_axis.append(i + 4)
            permutation_axis.append(i + 1)
        else:
            raise NotImplementedError
    permutation_axis.append(7)
    data = data.reshape(B, T // cuboid_size[0], H // cuboid_size[1], W // cuboid_size[2],
                        cuboid_size[0], cuboid_size[1], cuboid_size[2], C)
    data = data.permute(permutation_axis)
    data = data.reshape((B, T, H, W, C))
    return data


class CuboidSelfAttentionLayer(nn.Module):
    """Implements the cuboid self attention.

    The idea of Cuboid Self Attention is to divide the input tensor (T, H, W) into several non-overlapping cuboids.
    We apply self-attention inside each cuboid and all cuboid-level self attentions are executed in parallel.

    We adopt two mechanisms for decomposing the input tensor into cuboids:

    1) local:
        We group the tensors within a local window, e.g., X[t:(t+b_t), h:(h+b_h), w:(w+b_w)]. We can also apply the
        shifted window strategy proposed in "[ICCV2021] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
    2) dilated:
        Inspired by the success of dilated convolution "[ICLR2016] Multi-Scale Context Aggregation by Dilated Convolutions",
         we split the tensor with dilation factors that are tied to the size of the cuboid. For example, for a cuboid that has width `b_w`,
         we sample the elements starting from 0 as 0, w / b_w, 2 * w / b_w, ..., (b_w - 1) * w / b_w.

    The cuboid attention can be viewed as a generalization of the attention mechanism proposed in Video Swin Transformer, https://arxiv.org/abs/2106.13230.
    The computational complexity of CuboidAttention can be simply calculated as O(T H W * b_t b_h b_w). To cover multiple correlation patterns,
    we are able to combine multiple CuboidAttention layers with different configurations such as cuboid size, shift size, and local / global decomposing strategy.

    In addition, it is straight-forward to extend the cuboid attention to other types of spatiotemporal data that are not described
    as regular tensors. We need to define alternative approaches to partition the data into "cuboids".

    In addition, inspired by "[NeurIPS2021] Do Transformers Really Perform Badly for Graph Representation?",
     "[NeurIPS2020] Big Bird: Transformers for Longer Sequences", "[EMNLP2021] Longformer: The Long-Document Transformer", we keep
     $K$ global vectors to record the global status of the spatiotemporal system. These global vectors will attend to the whole tensor and
     the vectors inside each individual cuboids will also attend to the global vectors so that they can peep into the global status of the system.

    """
    def __init__(self,
                 dim,
                 num_heads,
                 cuboid_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 strategy=('l', 'l', 'l'),
                 padding_type='ignore',
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_final_proj=True,
                 norm_layer='layer_norm',
                 use_global_vector=False,
                 use_global_self_attn=False,
                 separate_global_qkv=False,
                 global_dim_ratio=1,
                 checkpoint_level=True,
                 use_relative_pos=True,
                 attn_linear_init_mode="0",
                 ffn_linear_init_mode="0",
                 norm_init_mode="0",
                 ):
        """

        Parameters
        ----------
        dim
            The dimension of the input tensor
        num_heads
            The number of heads
        cuboid_size
            The size of each cuboid
        shift_size
            The size for shifting the windows.
        strategy
            The decomposition strategy of the tensor. 'l' stands for local and 'd' stands for dilated.
        padding_type
            The type of padding.
        qkv_bias
            Whether to enable bias in calculating qkv attention
        qk_scale
            Whether to enable scale factor when calculating the attention.
        attn_drop
            The attention dropout
        proj_drop
            The projection dropout
        use_final_proj
            Whether to use the final projection or not
        norm_layer
            The normalization layer
        use_global_vector
            Whether to use the global vector or not.
        use_global_self_attn
            Whether to do self attention among global vectors
        separate_global_qkv
            Whether to different network to calc q_global, k_global, v_global
        global_dim_ratio
            The dim (channels) of global vectors is `global_dim_ratio*dim`.
        checkpoint_level
            Whether to enable gradient checkpointing.
        """
        super(CuboidSelfAttentionLayer, self).__init__()
        # initialization
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.norm_init_mode = norm_init_mode

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.cuboid_size = cuboid_size
        self.shift_size = shift_size
        self.strategy = strategy
        self.padding_type = padding_type
        self.use_final_proj = use_final_proj
        self.use_relative_pos = use_relative_pos
        # global vectors
        self.use_global_vector = use_global_vector
        self.use_global_self_attn = use_global_self_attn
        self.separate_global_qkv = separate_global_qkv
        if global_dim_ratio != 1:
            assert separate_global_qkv == True, \
                f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio

        assert self.padding_type in ['ignore', 'zeros', 'nearest']
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if use_relative_pos:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * cuboid_size[0] - 1) * (2 * cuboid_size[1] - 1) * (2 * cuboid_size[2] - 1), num_heads))
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

            coords_t = torch.arange(self.cuboid_size[0])
            coords_h = torch.arange(self.cuboid_size[1])
            coords_w = torch.arange(self.cuboid_size[2])
            coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w))  # 3, Bt, Bh, Bw

            coords_flatten = torch.flatten(coords, 1)  # 3, Bt*Bh*Bw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Bt*Bh*Bw, Bt*Bh*Bw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Bt*Bh*Bw, Bt*Bh*Bw, 3
            relative_coords[:, :, 0] += self.cuboid_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.cuboid_size[1] - 1
            relative_coords[:, :, 2] += self.cuboid_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.cuboid_size[1] - 1) * (2 * self.cuboid_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * self.cuboid_size[2] - 1)
            relative_position_index = relative_coords.sum(-1)  # shape is (cuboid_volume, cuboid_volume)
            self.register_buffer("relative_position_index", relative_position_index)
        
        # 时空分离的交叉注意力组件
        # 时间维度的交叉注意力
        self.temporal_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.temporal_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.temporal_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 空间维度的多尺度交叉注意力
        self.spatial_scales = [1, 2, 4]  # 多尺度空间注意力
        self.spatial_attention_heads = nn.ModuleList([
            nn.ModuleDict({
                'q': nn.Linear(dim, dim, bias=qkv_bias),
                'k': nn.Linear(dim, dim, bias=qkv_bias),
                'v': nn.Linear(dim, dim, bias=qkv_bias),
            }) for _ in self.spatial_scales
        ])
        
        # 门控机制
        self.temporal_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.final_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.temporal_fusion = nn.Linear(dim, dim)
        self.spatial_fusion = nn.Linear(dim, dim)
        self.final_fusion = nn.Linear(dim * 2, dim)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if self.use_global_vector:
            if self.separate_global_qkv:
                self.l2g_q_net = nn.Linear(dim, dim, bias=qkv_bias)
                self.l2g_global_kv_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim * 2,
                    bias=qkv_bias)
                self.g2l_global_q_net = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=dim,
                    bias=qkv_bias)
                self.g2l_k_net = nn.Linear(
                    in_features=dim,
                    out_features=dim,
                    bias=qkv_bias)
                self.g2l_v_net = nn.Linear(
                    in_features=dim,
                    out_features=global_dim_ratio * dim,
                    bias=qkv_bias)
                if self.use_global_self_attn:
                    self.g2g_global_qkv_net = nn.Linear(
                        in_features=global_dim_ratio * dim,
                        out_features=global_dim_ratio * dim * 3,
                        bias=qkv_bias)
            else:
                self.global_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.global_attn_drop = nn.Dropout(attn_drop)

        if use_final_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

            if self.use_global_vector:
                self.global_proj = nn.Linear(
                    in_features=global_dim_ratio * dim,
                    out_features=global_dim_ratio * dim)

        self.norm = get_norm_layer(norm_layer, in_channels=dim)
        if self.use_global_vector:
            self.global_vec_norm = get_norm_layer(norm_layer,
                                                  in_channels=global_dim_ratio*dim)

        self.checkpoint_level = checkpoint_level
        self.reset_parameters()

    def reset_parameters(self):
        apply_initialization(self.qkv,
                             linear_mode=self.attn_linear_init_mode)
        
        # 初始化时空分离交叉注意力组件
        apply_initialization(self.temporal_q, linear_mode=self.attn_linear_init_mode)
        apply_initialization(self.temporal_k, linear_mode=self.attn_linear_init_mode)
        apply_initialization(self.temporal_v, linear_mode=self.attn_linear_init_mode)
        
        for spatial_head in self.spatial_attention_heads:
            apply_initialization(spatial_head['q'], linear_mode=self.attn_linear_init_mode)
            apply_initialization(spatial_head['k'], linear_mode=self.attn_linear_init_mode)
            apply_initialization(spatial_head['v'], linear_mode=self.attn_linear_init_mode)
        
        # 初始化门控机制
        for layer in self.temporal_gate:
            if isinstance(layer, nn.Linear):
                apply_initialization(layer, linear_mode=self.attn_linear_init_mode)
        for layer in self.spatial_gate:
            if isinstance(layer, nn.Linear):
                apply_initialization(layer, linear_mode=self.attn_linear_init_mode)
        for layer in self.final_gate:
            if isinstance(layer, nn.Linear):
                apply_initialization(layer, linear_mode=self.attn_linear_init_mode)
        
        # 初始化特征融合
        apply_initialization(self.temporal_fusion, linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.spatial_fusion, linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.final_fusion, linear_mode=self.ffn_linear_init_mode)
        
        if self.use_final_proj:
            apply_initialization(self.proj,
                                 linear_mode=self.ffn_linear_init_mode)
        apply_initialization(self.norm,
                             norm_mode=self.norm_init_mode)
        if self.use_global_vector:
            if self.separate_global_qkv:
                apply_initialization(self.l2g_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.l2g_global_kv_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_global_q_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_k_net,
                                     linear_mode=self.attn_linear_init_mode)
                apply_initialization(self.g2l_v_net,
                                     linear_mode=self.attn_linear_init_mode)
                if self.use_global_self_attn:
                    apply_initialization(self.g2g_global_qkv_net,
                                         linear_mode=self.attn_linear_init_mode)
            else:
                apply_initialization(self.global_qkv,
                                     linear_mode=self.attn_linear_init_mode)
            apply_initialization(self.global_vec_norm,
                                 norm_mode=self.norm_init_mode)

    def forward(self, x, x2, global_vectors=None):
        assert x.ndim == 5 and x2.ndim == 5, f"x shape: {x.shape}, x2 shape: {x2.shape}"
        # 输入：x torch.Size([2, 13, 32, 32, 128])
        x = self.norm(x) # torch.Size([2, 13, 32, 32, 128])
        x2 = self.norm(x2) # 对 x2 也进行归一化

        B, T, H, W, C_in = x.shape
        assert C_in == self.dim
        if self.use_global_vector and global_vectors is not None: # True
            _, num_global, _ = global_vectors.shape
            global_vectors = self.global_vec_norm(global_vectors) # torch.Size([2, 8, 128])

        cuboid_size, shift_size = update_cuboid_size_shift_size((T, H, W), self.cuboid_size,
                                                                self.shift_size, self.strategy) # [13,1,1], [0,0,0]
        # Step-1: Pad the input
        pad_t = (cuboid_size[0] - T % cuboid_size[0]) % cuboid_size[0] # 0
        pad_h = (cuboid_size[1] - H % cuboid_size[1]) % cuboid_size[1] # 0
        pad_w = (cuboid_size[2] - W % cuboid_size[2]) % cuboid_size[2] # 0

        # We use generalized padding
        x = _generalize_padding(x, pad_t, pad_h, pad_w, self.padding_type) # torch.Size([2, 13, 32, 32, 128])
        x2 = _generalize_padding(x2, pad_t, pad_h, pad_w, self.padding_type) # 对 x2 也进行填充

        # Step-2: Shift the tensor based on shift window attention.

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            shifted_x2 = torch.roll(x2, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x # torch.Size([2, 13, 32, 32, 128])
            shifted_x2 = x2
        # Step-3: Reorder the tensor
        # (B, num_cuboids, cuboid_volume, C)
        reordered_x = cuboid_reorder(shifted_x, cuboid_size=cuboid_size, strategy=self.strategy) # torch.Size([2, 1024, 13, 128])
        reordered_x2 = cuboid_reorder(shifted_x2, cuboid_size=cuboid_size, strategy=self.strategy) # 对 x2 也进行重排序
        _, num_cuboids, cuboid_volume, _ = reordered_x.shape
        # Step-4: Perform cross-attention between x and x2
        # (num_cuboids, cuboid_volume, cuboid_volume)
        attn_mask = compute_cuboid_self_attention_mask((T, H, W), cuboid_size,
                                                       shift_size=shift_size,
                                                       strategy=self.strategy,
                                                       padding_type=self.padding_type,
                                                       device=x.device) # torch.Size([1024, 13, 13])
        head_C = C_in // self.num_heads
        
        # ========== 时空分离的交叉注意力 ==========
        
        # 1. 时间维度的交叉注意力
        # 将数据重塑为时间维度优先的形式
        x_temp = reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in)
        x2_temp = reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in)
        
        # 计算时间维度的 Q, K, V
        q_temp_x = self.temporal_q(x_temp)  # (B, num_cuboids, cuboid_volume, C)
        k_temp_x2 = self.temporal_k(x2_temp)  # (B, num_cuboids, cuboid_volume, C)
        v_temp_x2 = self.temporal_v(x2_temp)  # (B, num_cuboids, cuboid_volume, C)
        
        # 时间维度交叉注意力
        q_temp_x = q_temp_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        k_temp_x2 = k_temp_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        v_temp_x2 = v_temp_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        
        temp_attn_score = (q_temp_x * self.scale) @ k_temp_x2.transpose(-2, -1)
        # 修复注意力掩码维度：需要扩展到多头维度
        temp_attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, num_cuboids, cuboid_volume, cuboid_volume)
        temp_attn_weights = masked_softmax(temp_attn_score, mask=temp_attn_mask)
        temp_attn_weights = self.attn_drop(temp_attn_weights)
        temp_out = (temp_attn_weights @ v_temp_x2).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, C_in)
        
        # 时间维度门控
        temp_gate = self.temporal_gate(torch.cat([x_temp, temp_out], dim=-1))
        temp_out = temp_gate * temp_out
        temp_out = self.temporal_fusion(temp_out)
        
        # 2. 空间维度的多尺度交叉注意力
        spatial_outputs = []
        for scale_idx, scale in enumerate(self.spatial_scales):
            # 对空间维度进行下采样
            if scale > 1:
                # 重塑为空间维度进行下采样
                x_spatial = reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in)
                x2_spatial = reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in)
                
                # 正确处理空间维度：只对cuboid内的空间维度(bH, bW)进行池化
                # cuboid_volume = bT * bH * bW，其中bT=13, bH=1, bW=1
                bT, bH, bW = cuboid_size
                
                # 重塑为 (B, num_cuboids, bT, bH, bW, C_in)
                x_spatial_reshaped = x_spatial.reshape(B, num_cuboids, bT, bH, bW, C_in)
                x2_spatial_reshaped = x2_spatial.reshape(B, num_cuboids, bT, bH, bW, C_in)
                
                # 对空间维度(bH, bW)进行下采样
                if bH > 1 or bW > 1:
                    # 重塑为 (B * num_cuboids * bT, bH, bW, C_in) 然后进行2D池化
                    x_spatial_2d = x_spatial_reshaped.reshape(B * num_cuboids * bT, bH, bW, C_in).permute(0, 3, 1, 2)
                    x2_spatial_2d = x2_spatial_reshaped.reshape(B * num_cuboids * bT, bH, bW, C_in).permute(0, 3, 1, 2)
                    
                    # 计算新的空间尺寸
                    new_bH = max(1, bH // scale)
                    new_bW = max(1, bW // scale)
                    
                    # 只有当空间尺寸大于1时才进行池化
                    if bH > 1 and bW > 1:
                        x_spatial_down = F.avg_pool2d(x_spatial_2d, kernel_size=scale, stride=scale)
                        x2_spatial_down = F.avg_pool2d(x2_spatial_2d, kernel_size=scale, stride=scale)
                    elif bH > 1:
                        x_spatial_down = F.avg_pool2d(x_spatial_2d, kernel_size=(scale, 1), stride=(scale, 1))
                        x2_spatial_down = F.avg_pool2d(x2_spatial_2d, kernel_size=(scale, 1), stride=(scale, 1))
                    elif bW > 1:
                        x_spatial_down = F.avg_pool2d(x_spatial_2d, kernel_size=(1, scale), stride=(1, scale))
                        x2_spatial_down = F.avg_pool2d(x2_spatial_2d, kernel_size=(1, scale), stride=(1, scale))
                    else:
                        # 如果bH=1, bW=1，则不需要下采样
                        x_spatial_down = x_spatial_2d
                        x2_spatial_down = x2_spatial_2d
                    
                    # 重塑回 (B, num_cuboids, bT, new_bH, new_bW, C_in)
                    x_spatial_down = x_spatial_down.permute(0, 2, 3, 1).reshape(B, num_cuboids, bT, new_bH, new_bW, C_in)
                    x2_spatial_down = x2_spatial_down.permute(0, 2, 3, 1).reshape(B, num_cuboids, bT, new_bH, new_bW, C_in)
                    
                    # 重塑为 (B, num_cuboids, new_cuboid_volume, C_in)
                    new_cuboid_volume = bT * new_bH * new_bW
                    x_spatial_down = x_spatial_down.reshape(B, num_cuboids, new_cuboid_volume, C_in)
                    x2_spatial_down = x2_spatial_down.reshape(B, num_cuboids, new_cuboid_volume, C_in)
                else:
                    # 如果bH=1, bW=1，则不需要下采样
                    x_spatial_down = x_spatial
                    x2_spatial_down = x2_spatial
            else:
                x_spatial_down = reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in)
                x2_spatial_down = reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in)
            
            # 计算空间维度的 Q, K, V
            spatial_head = self.spatial_attention_heads[scale_idx]
            q_spatial_x = spatial_head['q'](x_spatial_down)
            k_spatial_x2 = spatial_head['k'](x2_spatial_down)
            v_spatial_x2 = spatial_head['v'](x2_spatial_down)
            
            # 获取当前cuboid_volume（可能因为下采样而改变）
            current_cuboid_volume = x_spatial_down.shape[2]
            
            # 空间维度交叉注意力
            q_spatial_x = q_spatial_x.reshape(B, num_cuboids, current_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            k_spatial_x2 = k_spatial_x2.reshape(B, num_cuboids, current_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            v_spatial_x2 = v_spatial_x2.reshape(B, num_cuboids, current_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            
            spatial_attn_score = (q_spatial_x * self.scale) @ k_spatial_x2.transpose(-2, -1)
            # 修复注意力掩码维度：需要扩展到多头维度，并调整到当前cuboid_volume
            if current_cuboid_volume == cuboid_volume:
                spatial_attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, num_cuboids, current_cuboid_volume, current_cuboid_volume)
            else:
                # 如果cuboid_volume改变了，需要调整注意力掩码
                spatial_attn_mask = torch.ones(B, self.num_heads, num_cuboids, current_cuboid_volume, current_cuboid_volume, device=x.device)
            
            spatial_attn_weights = masked_softmax(spatial_attn_score, mask=spatial_attn_mask)
            spatial_attn_weights = self.attn_drop(spatial_attn_weights)
            spatial_out = (spatial_attn_weights @ v_spatial_x2).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, current_cuboid_volume, C_in)
            
            # 如果进行了下采样，需要上采样回原始尺寸
            if scale > 1 and current_cuboid_volume != cuboid_volume:
                # 重塑为 (B, num_cuboids, bT, new_bH, new_bW, C_in)
                bT, bH, bW = cuboid_size
                new_bH = max(1, bH // scale)
                new_bW = max(1, bW // scale)
                
                spatial_out_reshaped = spatial_out.reshape(B, num_cuboids, bT, new_bH, new_bW, C_in)
                
                if bH > 1 or bW > 1:
                    # 重塑为 (B * num_cuboids * bT, new_bH, new_bW, C_in) 然后进行2D上采样
                    spatial_out_2d = spatial_out_reshaped.reshape(B * num_cuboids * bT, new_bH, new_bW, C_in).permute(0, 3, 1, 2)
                    
                    spatial_out_upsampled = F.interpolate(
                        spatial_out_2d,
                        size=(bH, bW),
                        mode='nearest'
                    )
                    
                    # 重塑回 (B, num_cuboids, bT, bH, bW, C_in)
                    spatial_out_reshaped = spatial_out_upsampled.permute(0, 2, 3, 1).reshape(B, num_cuboids, bT, bH, bW, C_in)
                
                # 重塑为 (B, num_cuboids, cuboid_volume, C_in)
                spatial_out = spatial_out_reshaped.reshape(B, num_cuboids, cuboid_volume, C_in)
            
            spatial_outputs.append(spatial_out)
        
        # 融合多尺度空间注意力结果
        spatial_out = sum(spatial_outputs) / len(spatial_outputs)
        
        # 空间维度门控
        spatial_gate = self.spatial_gate(torch.cat([reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in), spatial_out], dim=-1))
        spatial_out = spatial_gate * spatial_out
        spatial_out = self.spatial_fusion(spatial_out)
        
        # 3. 融合时空特征
        combined_out = self.final_fusion(torch.cat([temp_out, spatial_out], dim=-1))
        
        # 最终门控机制
        final_gate = self.final_gate(torch.cat([reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in), combined_out], dim=-1))
        final_out = final_gate * combined_out
        
        # 重塑回原始形状
        reordered_x = final_out.reshape(B, num_cuboids, cuboid_volume, C_in)
        
        # 对 x2 进行相同的处理（对称的交叉注意力）
        # 时间维度的交叉注意力（x2 使用 x 的信息）
        q_temp_x2 = self.temporal_q(x2_temp)
        k_temp_x = self.temporal_k(x_temp)
        v_temp_x = self.temporal_v(x_temp)
        
        q_temp_x2 = q_temp_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        k_temp_x = k_temp_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        v_temp_x = v_temp_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
        
        temp_attn_score2 = (q_temp_x2 * self.scale) @ k_temp_x.transpose(-2, -1)
        temp_attn_weights2 = masked_softmax(temp_attn_score2, mask=attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, num_cuboids, cuboid_volume, cuboid_volume))
        temp_attn_weights2 = self.attn_drop(temp_attn_weights2)
        temp_out2 = (temp_attn_weights2 @ v_temp_x).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, C_in)
        
        temp_gate2 = self.temporal_gate(torch.cat([x2_temp, temp_out2], dim=-1))
        temp_out2 = temp_gate2 * temp_out2
        temp_out2 = self.temporal_fusion(temp_out2)
        
        # 空间维度的多尺度交叉注意力（x2 使用 x 的信息）
        spatial_outputs2 = []
        for scale_idx, scale in enumerate(self.spatial_scales):
            if scale > 1:
                # 正确处理空间维度：只对cuboid内的空间维度(bH, bW)进行池化
                # cuboid_volume = bT * bH * bW，其中bT=13, bH=1, bW=1
                bT, bH, bW = cuboid_size
                
                # 计算新的空间尺寸
                new_bH = max(1, bH // scale)
                new_bW = max(1, bW // scale)
                
                # 重塑为 (B, num_cuboids, bT, bH, bW, C_in)
                x2_spatial_reshaped = x2_spatial.reshape(B, num_cuboids, bT, bH, bW, C_in)
                x_spatial_reshaped = x_spatial.reshape(B, num_cuboids, bT, bH, bW, C_in)
                
                # 对空间维度(bH, bW)进行下采样
                if bH > 1 or bW > 1:
                    # 重塑为 (B * num_cuboids * bT, bH, bW, C_in) 然后进行2D池化
                    x2_spatial_2d = x2_spatial_reshaped.reshape(B * num_cuboids * bT, bH, bW, C_in).permute(0, 3, 1, 2)
                    x_spatial_2d = x_spatial_reshaped.reshape(B * num_cuboids * bT, bH, bW, C_in).permute(0, 3, 1, 2)
                    
                    # 只有当空间尺寸大于1时才进行池化
                    if bH > 1 and bW > 1:
                        x2_spatial_down = F.avg_pool2d(x2_spatial_2d, kernel_size=scale, stride=scale)
                        x_spatial_down_x = F.avg_pool2d(x_spatial_2d, kernel_size=scale, stride=scale)
                    elif bH > 1:
                        x2_spatial_down = F.avg_pool2d(x2_spatial_2d, kernel_size=(scale, 1), stride=(scale, 1))
                        x_spatial_down_x = F.avg_pool2d(x_spatial_2d, kernel_size=(scale, 1), stride=(scale, 1))
                    elif bW > 1:
                        x2_spatial_down = F.avg_pool2d(x2_spatial_2d, kernel_size=(1, scale), stride=(1, scale))
                        x_spatial_down_x = F.avg_pool2d(x_spatial_2d, kernel_size=(1, scale), stride=(1, scale))
                    else:
                        # 如果bH=1, bW=1，则不需要下采样
                        x2_spatial_down = x2_spatial_2d
                        x_spatial_down_x = x_spatial_2d
                else:
                    # 如果bH=1, bW=1，则不需要下采样
                    x2_spatial_down = reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in)
                    x_spatial_down_x = reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in)
                 
                # 重塑回 (B, num_cuboids, bT, new_bH, new_bW, C_in)
                x2_spatial_down = x2_spatial_down.permute(0, 2, 3, 1).reshape(B, num_cuboids, bT, new_bH, new_bW, C_in)
                x_spatial_down_x = x_spatial_down_x.permute(0, 2, 3, 1).reshape(B, num_cuboids, bT, new_bH, new_bW, C_in)
                 
                # 重塑为 (B, num_cuboids, new_cuboid_volume, C_in)
                new_cuboid_volume = bT * new_bH * new_bW
                x2_spatial_down = x2_spatial_down.reshape(B, num_cuboids, new_cuboid_volume, C_in)
                x_spatial_down_x = x_spatial_down_x.reshape(B, num_cuboids, new_cuboid_volume, C_in)
            else:
                x2_spatial_down = reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in)
                x_spatial_down_x = reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in)
            
            spatial_head = self.spatial_attention_heads[scale_idx]
            q_spatial_x2 = spatial_head['q'](x2_spatial_down)
            k_spatial_x = spatial_head['k'](x_spatial_down_x)
            v_spatial_x = spatial_head['v'](x_spatial_down_x)
            
            # 获取当前cuboid_volume（可能因为下采样而改变）
            current_cuboid_volume = x2_spatial_down.shape[2]
            
            q_spatial_x2 = q_spatial_x2.reshape(B, num_cuboids, current_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            k_spatial_x = k_spatial_x.reshape(B, num_cuboids, current_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            v_spatial_x = v_spatial_x.reshape(B, num_cuboids, current_cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            
            spatial_attn_score2 = (q_spatial_x2 * self.scale) @ k_spatial_x.transpose(-2, -1)
            # 修复注意力掩码维度：需要扩展到多头维度，并调整到当前cuboid_volume
            if current_cuboid_volume == cuboid_volume:
                spatial_attn_mask2 = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, num_cuboids, current_cuboid_volume, current_cuboid_volume)
            else:
                # 如果cuboid_volume改变了，需要调整注意力掩码
                spatial_attn_mask2 = torch.ones(B, self.num_heads, num_cuboids, current_cuboid_volume, current_cuboid_volume, device=x.device)
            
            spatial_attn_weights2 = masked_softmax(spatial_attn_score2, mask=spatial_attn_mask2)
            spatial_attn_weights2 = self.attn_drop(spatial_attn_weights2)
            spatial_out2 = (spatial_attn_weights2 @ v_spatial_x).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, current_cuboid_volume, C_in)
            
            if scale > 1 and current_cuboid_volume != cuboid_volume:
                # 重塑为 (B, num_cuboids, bT, new_bH, new_bW, C_in)
                bT, bH, bW = cuboid_size
                new_bH = max(1, bH // scale)
                new_bW = max(1, bW // scale)
                
                spatial_out2_reshaped = spatial_out2.reshape(B, num_cuboids, bT, new_bH, new_bW, C_in)
                
                if bH > 1 or bW > 1:
                    # 重塑为 (B * num_cuboids * bT, new_bH, new_bW, C_in) 然后进行2D上采样
                    spatial_out2_2d = spatial_out2_reshaped.reshape(B * num_cuboids * bT, new_bH, new_bW, C_in).permute(0, 3, 1, 2)
                    
                    spatial_out2_upsampled = F.interpolate(
                        spatial_out2_2d,
                        size=(bH, bW),
                        mode='nearest'
                    )
                    
                    # 重塑回 (B, num_cuboids, bT, bH, bW, C_in)
                    spatial_out2_reshaped = spatial_out2_upsampled.permute(0, 2, 3, 1).reshape(B, num_cuboids, bT, bH, bW, C_in)
                
                # 重塑为 (B, num_cuboids, cuboid_volume, C_in)
                spatial_out2 = spatial_out2_reshaped.reshape(B, num_cuboids, cuboid_volume, C_in)
            
            spatial_outputs2.append(spatial_out2)
        
        spatial_out2 = sum(spatial_outputs2) / len(spatial_outputs2)
        
        spatial_gate2 = self.spatial_gate(torch.cat([reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in), spatial_out2], dim=-1))
        spatial_out2 = spatial_gate2 * spatial_out2
        spatial_out2 = self.spatial_fusion(spatial_out2)
        
        combined_out2 = self.final_fusion(torch.cat([temp_out2, spatial_out2], dim=-1))
        
        final_gate2 = self.final_gate(torch.cat([reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in), combined_out2], dim=-1))
        final_out2 = final_gate2 * combined_out2
        
        reordered_x2 = final_out2.reshape(B, num_cuboids, cuboid_volume, C_in)
        
        # ========== 全局向量处理（保持原有逻辑） ==========
        # Calculate the local to global attention
        if self.use_global_vector and global_vectors is not None:
            global_head_C = self.global_dim_ratio * head_C
            if self.separate_global_qkv:
                l2g_q_x = self.l2g_q_net(reordered_x)\
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)\
                    .permute(0, 3, 1, 2, 4)
                l2g_q_x = l2g_q_x * self.scale
                l2g_q_x2 = self.l2g_q_net(reordered_x2)\
                    .reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C)\
                    .permute(0, 3, 1, 2, 4)
                l2g_q_x2 = l2g_q_x2 * self.scale
                l2g_global_kv = self.l2g_global_kv_net(global_vectors)\
                    .reshape(B, 1, num_global, 2, self.num_heads, head_C)\
                    .permute(3, 0, 4, 1, 2, 5)
                l2g_global_k, l2g_global_v = l2g_global_kv[0], l2g_global_kv[1]
                g2l_global_q = self.g2l_global_q_net(global_vectors)\
                    .reshape(B, num_global, self.num_heads, head_C)\
                    .permute(0, 2, 1, 3)
                g2l_global_q = g2l_global_q * self.scale
                # g2l_k和g2l_v应该用cuboid特征，不是global特征
                g2l_k = reordered_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                g2l_v = reordered_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                g2l_k2 = reordered_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                g2l_v2 = reordered_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                if self.use_global_self_attn:
                    g2g_global_qkv = self.g2g_global_qkv_net(global_vectors)\
                    .reshape(B, 1, num_global, 3, self.num_heads, global_head_C)\
                    .permute(3, 0, 4, 1, 2, 5)
                    g2g_global_q, g2g_global_k, g2g_global_v = g2g_global_qkv[0], g2g_global_qkv[1], g2g_global_qkv[2]
                    g2g_global_q = g2g_global_q.squeeze(2) * self.scale
            else:
                # 直接用cuboid特征做query
                l2g_q_x = reordered_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4) * self.scale
                l2g_q_x2 = reordered_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4) * self.scale
                q_global, k_global, v_global = self.global_qkv(global_vectors)\
                    .reshape(B, 1, num_global, 3, self.num_heads, head_C)\
                    .permute(3, 0, 4, 1, 2, 5)
                l2g_global_k, l2g_global_v = k_global, v_global
                g2l_global_q = q_global.squeeze(2) * self.scale
                # g2l_k和g2l_v应该用cuboid特征，不是global特征
                g2l_k = reordered_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                g2l_v = reordered_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                g2l_k2 = reordered_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                g2l_v2 = reordered_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
                if self.use_global_self_attn:
                    g2g_global_q, g2g_global_k, g2g_global_v = q_global.squeeze(2) * self.scale, k_global, v_global
            l2g_attn_score_x = l2g_q_x @ l2g_global_k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, N)
            l2g_attn_score_x2 = l2g_q_x2 @ l2g_global_k.transpose(-2, -1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, N)
            # 创建虚拟的局部到局部注意力分数（因为我们已经通过时空分离交叉注意力处理了）
            dummy_l2l_score = torch.zeros(B, self.num_heads, num_cuboids, cuboid_volume, cuboid_volume, device=x.device)
            attn_score_x_l2l_l2g = torch.cat((dummy_l2l_score, l2g_attn_score_x),
                                           dim=-1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume + N)
            attn_score_x2_l2l_l2g = torch.cat((dummy_l2l_score, l2g_attn_score_x2),
                                           dim=-1)  # Shape (B, num_heads, num_cuboids, cuboid_volume, cuboid_volume + N)
            attn_mask_l2l_l2g = F.pad(attn_mask, (0, num_global), "constant", 1) # torch.Size([1024, 13, 21])
            
            # 修复维度拼接问题：需要将局部特征和全局特征都转换为相同的维度
            # 局部特征: (B, num_heads, num_cuboids, cuboid_volume, C_in)
            local_feat_x = reordered_x2.reshape(B, num_cuboids, cuboid_volume, C_in).unsqueeze(1).expand(B, self.num_heads, num_cuboids, cuboid_volume, C_in)
            local_feat_x2 = reordered_x.reshape(B, num_cuboids, cuboid_volume, C_in).unsqueeze(1).expand(B, self.num_heads, num_cuboids, cuboid_volume, C_in)
            
            # 全局特征: (B, num_heads, num_cuboids, num_global, head_C) -> 需要转换为 (B, num_heads, num_cuboids, num_global, C_in)
            global_feat = l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)
            # 将全局特征从 head_C 维度扩展到 C_in 维度
            global_feat_expanded = global_feat.unsqueeze(-1).expand(B, self.num_heads, num_cuboids, num_global, head_C, self.num_heads).reshape(B, self.num_heads, num_cuboids, num_global, C_in)
            
            v_l_g = torch.cat((local_feat_x, global_feat_expanded), dim=3) # (B, num_heads, num_cuboids, cuboid_volume + num_global, C_in)
            v_l_g2 = torch.cat((local_feat_x2, global_feat_expanded), dim=3) # (B, num_heads, num_cuboids, cuboid_volume + num_global, C_in)
            # local to local and global attention
            attn_score_x_l2l_l2g = masked_softmax(attn_score_x_l2l_l2g, mask=attn_mask_l2l_l2g) # torch.Size([2, 4, 1024, 13, 21])
            attn_score_x2_l2l_l2g = masked_softmax(attn_score_x2_l2l_l2g, mask=attn_mask_l2l_l2g) # torch.Size([2, 4, 1024, 13, 21])
            attn_score_x_l2l_l2g = self.attn_drop(attn_score_x_l2l_l2g)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, mem_cuboid_volume + K)) #torch.Size([2, 4, 1024, 13, 21])
            attn_score_x2_l2l_l2g = self.attn_drop(attn_score_x2_l2l_l2g)  # Shape (B, num_heads, num_cuboids, x_cuboid_volume, mem_cuboid_volume + K))
            # 拼接时，local和global都用head_C维度
            local_feat_x = reordered_x2.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)  # (B, num_heads, num_cuboids, cuboid_volume, head_C)
            local_feat_x2 = reordered_x.reshape(B, num_cuboids, cuboid_volume, self.num_heads, head_C).permute(0, 3, 1, 2, 4)
            global_feat = l2g_global_v.expand(B, self.num_heads, num_cuboids, num_global, head_C)  # (B, num_heads, num_cuboids, num_global, head_C)
            v_l_g = torch.cat((local_feat_x, global_feat), dim=3)  # (B, num_heads, num_cuboids, cuboid_volume+num_global, head_C)
            v_l_g2 = torch.cat((local_feat_x2, global_feat), dim=3)

            # local to local and global attention
            attn_score_x_l2l_l2g = masked_softmax(attn_score_x_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_x2_l2l_l2g = masked_softmax(attn_score_x2_l2l_l2g, mask=attn_mask_l2l_l2g)
            attn_score_x_l2l_l2g = self.attn_drop(attn_score_x_l2l_l2g)
            attn_score_x2_l2l_l2g = self.attn_drop(attn_score_x2_l2l_l2g)
            # (B, num_heads, num_cuboids, cuboid_volume, head_C) -> (B, num_cuboids, cuboid_volume, self.dim)
            reordered_x = (attn_score_x_l2l_l2g @ v_l_g).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)
            reordered_x2 = (attn_score_x2_l2l_l2g @ v_l_g2).permute(0, 2, 3, 1, 4).reshape(B, num_cuboids, cuboid_volume, self.dim)
            # update global vectors
            if self.padding_type == 'ignore': # 没走这里
                g2l_attn_mask = torch.ones((1, T, H, W, 1), device=x.device)
                if pad_t > 0 or pad_h > 0 or pad_w > 0:
                    g2l_attn_mask = F.pad(g2l_attn_mask, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
                if any(i > 0 for i in shift_size):
                    g2l_attn_mask = torch.roll(g2l_attn_mask, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                                               dims=(1, 2, 3))
                g2l_attn_mask = g2l_attn_mask.reshape((-1,))
            else:
                g2l_attn_mask = None
            g2l_attn_score = g2l_global_q @ g2l_k.reshape(B, self.num_heads, num_cuboids * cuboid_volume, head_C).transpose(-2, -1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume)
            g2l_attn_score2 = g2l_global_q @ g2l_k2.reshape(B, self.num_heads, num_cuboids * cuboid_volume, head_C).transpose(-2, -1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume)
            if self.use_global_self_attn:
                g2g_attn_score = g2g_global_q @ g2g_global_k.squeeze(2).transpose(-2, -1)
                g2all_attn_score = torch.cat((g2l_attn_score, g2g_attn_score),
                                             dim=-1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume + N)
                g2all_attn_score2 = torch.cat((g2l_attn_score2, g2g_attn_score),
                                             dim=-1)  # Shape (B, num_heads, N, num_cuboids * cuboid_volume + N)
                if g2l_attn_mask is not None:
                    g2all_attn_mask = F.pad(g2l_attn_mask, (0, num_global), "constant", 1)
                else:
                    g2all_attn_mask = None
                new_v = torch.cat((g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C),
                                   g2g_global_v.reshape(B, self.num_heads, num_global, global_head_C)),
                                  dim=2)
                new_v2 = torch.cat((g2l_v2.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C),
                                   g2g_global_v.reshape(B, self.num_heads, num_global, global_head_C)),
                                  dim=2)
            else:
                g2all_attn_score = g2l_attn_score
                g2all_attn_score2 = g2l_attn_score2
                g2all_attn_mask = g2l_attn_mask
                new_v = g2l_v.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C)
                new_v2 = g2l_v2.reshape(B, self.num_heads, num_cuboids * cuboid_volume, global_head_C)
            g2all_attn_score = masked_softmax(g2all_attn_score, mask=g2all_attn_mask)
            g2all_attn_score2 = masked_softmax(g2all_attn_score2, mask=g2all_attn_mask)
            g2all_attn_score = self.global_attn_drop(g2all_attn_score)
            g2all_attn_score2 = self.global_attn_drop(g2all_attn_score2)
            new_global_vector = (g2all_attn_score @ new_v).permute(0, 2, 1, 3).\
                reshape(B, num_global, self.global_dim_ratio*self.dim)
            new_global_vector2 = (g2all_attn_score2 @ new_v2).permute(0, 2, 1, 3).\
                reshape(B, num_global, self.global_dim_ratio*self.dim)
            # 合并两个全局向量
            new_global_vector = (new_global_vector + new_global_vector2) / 2

        if self.use_final_proj:
            reordered_x = self.proj_drop(self.proj(reordered_x))
            reordered_x2 = self.proj_drop(self.proj(reordered_x2))
            if self.use_global_vector:
                new_global_vector = self.proj_drop(self.global_proj(new_global_vector))
        # Step-5: Shift back and slice
        shifted_x = cuboid_reorder_reverse(reordered_x, cuboid_size=cuboid_size, strategy=self.strategy,
                                           orig_data_shape=(T + pad_t, H + pad_h, W + pad_w))
        shifted_x2 = cuboid_reorder_reverse(reordered_x2, cuboid_size=cuboid_size, strategy=self.strategy,
                                           orig_data_shape=(T + pad_t, H + pad_h, W + pad_w))
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            x2 = torch.roll(shifted_x2, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
            x2 = shifted_x2
        x = _generalize_unpadding(x, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        x2 = _generalize_unpadding(x2, pad_t=pad_t, pad_h=pad_h, pad_w=pad_w, padding_type=self.padding_type)
        if self.use_global_vector and global_vectors is not None:
            return x, x2, new_global_vector
        else:
            return x, x2
