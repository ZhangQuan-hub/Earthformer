# Methodology

Our proposed Cuboid Transformer architecture introduces a novel spatiotemporal prediction framework that effectively integrates multi-modal meteorological data through sophisticated cross-attention mechanisms. The methodology comprises three core components: information fusion for multi-modal data integration, temporal cross-attention for capturing long-term dependencies, and spatial multi-scale cross-attention for hierarchical spatial feature extraction.

## 3.1 Multi-Modal Information Fusion

The foundation of our approach lies in the effective fusion of heterogeneous meteorological data streams, specifically precipitation data (x₁) and lightning data (x₂) from the SEVIR dataset. This dual-stream architecture enables comprehensive weather pattern understanding by leveraging the complementary nature of these data modalities.

**Data Preprocessing and Alignment**: Both input streams x₁ ∈ ℝ^(B×T×H×W×C) and x₂ ∈ ℝ^(B×T×H×W×C) undergo identical spatial-temporal preprocessing to ensure dimensional consistency, where B represents batch size, T denotes temporal length, H and W are spatial dimensions, and C is the channel dimension. Layer normalization is applied to both streams independently to stabilize training dynamics.

**Cuboid Decomposition Strategy**: Following the cuboid attention paradigm, we partition the input tensor (T, H, W) into non-overlapping cuboids of size (b_t, b_h, b_w). The decomposition supports both local and dilated strategies, enabling flexible spatial-temporal locality modeling. For meteorological data, we primarily employ local decomposition with adaptive cuboid sizing based on the temporal resolution of weather phenomena.

**Initial Feature Alignment**: Before cross-modal interaction, both data streams are processed through shared embedding layers to project them into a unified feature space. This alignment ensures that precipitation and lightning features can be meaningfully compared and combined in subsequent attention computations.

## 3.2 Temporal Cross-Attention Mechanism

The temporal cross-attention component captures long-range temporal dependencies by enabling each time step to selectively attend to relevant information across the entire temporal sequence from the complementary data stream.

**Cross-Modal Query-Key-Value Computation**: For temporal attention, we compute queries Q_temp from x₁, while keys K_temp and values V_temp are derived from x₂, and vice versa. This asymmetric design allows each modality to actively seek relevant information from the other:

```
Q_temp^(1) = x₁W_q^temp,  K_temp^(2) = x₂W_k^temp,  V_temp^(2) = x₂W_v^temp
Q_temp^(2) = x₂W_q^temp,  K_temp^(1) = x₁W_k^temp,  V_temp^(1) = x₁W_v^temp
```

**Temporal Attention Computation**: The temporal cross-attention weights are computed as:

```
A_temp = softmax((Q_temp^(1) K_temp^(2)ᵀ)/√d_k)
```

where d_k is the key dimension. This allows each temporal position in the precipitation data to attend to all temporal positions in the lightning data, capturing cross-modal temporal correlations.

**Gated Fusion for Temporal Features**: To control information flow and prevent feature interference, we employ a learnable gating mechanism:

```
G_temp = σ(W_g[x₁; A_temp V_temp^(2)])
F_temp = G_temp ⊙ (W_f A_temp V_temp^(2))
```

where σ is the sigmoid function, [;] denotes concatenation, ⊙ represents element-wise multiplication, and W_g, W_f are learnable transformation matrices.

## 3.3 Spatial Multi-Scale Cross-Attention

The spatial multi-scale cross-attention mechanism addresses the challenge of capturing meteorological phenomena at different spatial scales, from local convective cells to large-scale weather systems.

**Multi-Scale Spatial Decomposition**: We implement a hierarchical spatial attention mechanism operating at multiple scales S = {1, 2, 4}. For each scale s ∈ S, spatial features are downsampled using adaptive pooling:

```
x₁^(s) = AdaptivePool(x₁, scale=s)
x₂^(s) = AdaptivePool(x₂, scale=s)
```

The pooling operation intelligently handles boundary conditions, skipping pooling when spatial dimensions (b_h, b_w) equal 1 to prevent degenerate cases.

**Scale-Specific Cross-Attention**: Each scale maintains dedicated query, key, and value projection networks to capture scale-appropriate feature representations:

```
Q_spatial^(s,1) = x₁^(s)W_q^(s),  K_spatial^(s,2) = x₂^(s)W_k^(s),  V_spatial^(s,2) = x₂^(s)W_v^(s)
```

The cross-attention computation at scale s is:

```
A_spatial^(s) = softmax((Q_spatial^(s,1) K_spatial^(s,2)ᵀ)/√d_k)
F_spatial^(s) = A_spatial^(s) V_spatial^(s,2)
```

**Upsampling and Feature Aggregation**: Features computed at different scales are upsampled back to the original spatial resolution using nearest-neighbor interpolation to preserve spatial coherence:

```
F_spatial^(s,↑) = Upsample(F_spatial^(s), size=(b_h, b_w))
```

The multi-scale features are then aggregated through weighted averaging:

```
F_spatial = (1/|S|) ∑_{s∈S} F_spatial^(s,↑)
```

**Spatial Gating and Integration**: Similar to temporal attention, spatial features undergo gated fusion to ensure selective information integration:

```
G_spatial = σ(W_g^spatial[x₁; F_spatial])
F_spatial^final = G_spatial ⊙ (W_f^spatial F_spatial)
```

**Final Feature Fusion**: The temporal and spatial cross-attention outputs are combined through a final fusion layer with gating control:

```
F_combined = W_fusion[F_temp; F_spatial^final]
G_final = σ(W_g^final[x₁; F_combined])
Output = G_final ⊙ F_combined + x₁
```

This residual connection preserves the original input information while enhancing it with cross-modal spatiotemporal features. The symmetric processing ensures that both x₁ and x₂ benefit from cross-modal attention, with the final outputs being F₁ and F₂ respectively.

The proposed architecture leverages global vector support for maintaining global spatiotemporal context, though this component serves as an auxiliary enhancement rather than a core innovation. Through this comprehensive methodology, our Cuboid Transformer effectively captures complex spatiotemporal dependencies in meteorological data while facilitating robust multi-modal information fusion for accurate weather prediction.
