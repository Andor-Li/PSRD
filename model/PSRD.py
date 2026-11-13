import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
class LayerNorm2d(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type == 'WithBias':
            self.body = WithBias_LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported LayerNorm_type: {LayerNorm_type}")
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class HybridNorm2d(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super().__init__()
        self.ln = LayerNorm2d(dim, LayerNorm_type="WithBias")
        self.inorm = nn.InstanceNorm2d(dim, affine=True)
        self.alpha_logit = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(init_alpha))))
        )
    def forward(self, x):
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * self.ln(x) + (1.0 - alpha) * self.inorm(x)
    
# =====================================================================
# 1. Noise-Robust Log-Derivative Prior Extractor（NR-LDPE）
# =====================================================================
class NR_LDPE(nn.Module):
    def __init__(self, eps=1e-4, window_size=3, alpha=10.0):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=window_size//2)

    def forward(self, x):
        # 1. 提取原始对数导数先验
        log_x = torch.log(torch.clamp(x, min=self.eps))
        log_x_pad = F.pad(log_x, (0, 0, 0, 0, 1, 0), mode='replicate')
        grad_x = log_x_pad[:, 1:, :, :] - log_x_pad[:, :-1, :, :]
        
        # 2. 局部方差估计 (感知不确定性)
        # Var(X) = E[X^2] - (E[X])^2
        mean_x = self.pool(x)
        mean_x2 = self.pool(x ** 2)
        variance = torch.clamp(mean_x2 - mean_x ** 2, min=0)
        
        # 将多光谱方差沿通道求均值，得到空间不确定性图
        spatial_var = torch.mean(variance, dim=1, keepdim=True)
        
        # 3. 计算不确定性掩码 U (0 到 1 之间)
        # 方差越大(噪声越强)，U 越接近 1
        U = 1.0 - torch.exp(-self.alpha * spatial_var)
        
        # 4. 动态调制物理先验 (在噪声区衰减导数幅度)
        robust_grad = grad_x * (1.0 - U)
        return F.instance_norm(robust_grad)

# =====================================================================
# 2. Prior-Guided Linear Attention (PGLA)
# =====================================================================
class PGLA(nn.Module):
    """
    O(N) 复杂度的物理先验引导线性注意力。
    通过通道拼接构造扩展特征：
        Q' = [phi(Q), sqrt(gamma) P]
        K' = [phi(K), sqrt(gamma) P]
    从而隐式建模核相似度：
        phi(Q)phi(K)^T + gamma * P P^T
    并以线性 attention 形式 O(N) 计算输出。
    注意：这不等价于标准 softmax(QK^T)V，而是等价于
    一个带先验附加核项的线性核注意力。
    """
    def __init__(self, dim, prior_channels=31):
        super().__init__()
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        
        self.prior_proj = nn.Conv2d(prior_channels, dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1)) # 可学习控制系数
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, illum_feat, prior_feat):
        B, C, H, W = illum_feat.shape
        N = H * W
        
        # 1. 生成 Q, K, V
        q = self.q(illum_feat).view(B, C, N).transpose(1, 2) # [B, N, C]
        k = self.k(illum_feat).view(B, C, N).transpose(1, 2) # [B, N, C]
        v = self.v(illum_feat).view(B, C, N).transpose(1, 2) # [B, N, C]
        
        # 2. Linear Attention 核函数映射 (保证正数)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # 3. 处理绝对物理先验
        p_feat = self.prior_proj(prior_feat).view(B, C, N).transpose(1, 2) # [B, N, C]
        p_feat = F.normalize(p_feat, dim=-1)
        
        # 4. 避免 N^2 计算
        sqrt_gamma = torch.sqrt(torch.clamp(self.gamma, min=1e-6))
        p_scaled = p_feat * sqrt_gamma
        
        q_prime = torch.cat([q, p_scaled], dim=-1) # [B, N, 2C]
        k_prime = torch.cat([k, p_scaled], dim=-1) # [B, N, 2C]
        
        # 5. O(N) 线性计算顺序: Q(K^T V)
        kv = torch.einsum('b n c, b n d -> b c d', k_prime, v) # [B, 2C, C]
        
        # 归一化分母
        z = 1.0 / (torch.einsum('b n c, b c -> b n', q_prime, k_prime.sum(dim=1)) + 1e-6)
        
        out = torch.einsum('b n c, b c d -> b n d', q_prime, kv) * z.unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return self.proj(out)

# =====================================================================
# 3. Content-Adaptive Complementary Frequency Decomposition (CA-CFD)
# =====================================================================
class DynamicFrequencyMask(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 利用一个轻量级的网络，从空间特征预测频域掩码的权重
        self.freq_weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x_spatial):
        B, C, H, W = x_spatial.shape
        # 预测当前场景的自适应截止阈值系数 (基础设为0.08，网络可动态调节上下限)
        adaptive_scale = self.freq_weight_predictor(x_spatial) * 0.1 + 0.03 # 范围 [0.03, 0.13]
       
        # 归一化坐标范围 [-1, 1]
        y = torch.linspace(-1, 1, H, device=x_spatial.device)
        x = torch.linspace(-1, 1, W, device=x_spatial.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        # 计算半径（归一化频率，最大为 sqrt(2) 约 1.414，但通常取最大 1）
        radius = torch.sqrt(grid_y**2 + grid_x**2)            # [H, W]

        # 生成平滑的软掩码 (类似巴特沃斯低通滤波器，杜绝振铃效应, n = 4 平滑过渡)
        n = 4
        # 将 adaptive_scale 扩展为 [B, 1, H, W] 以广播
        scale_expanded = adaptive_scale.view(B, 1, 1, 1)
        low_freq_soft_mask = 1.0 / (1.0 + (radius / (scale_expanded + 1e-6)) ** (2 * n))
        
        return low_freq_soft_mask
    
class CA_OIS(nn.Module):
    """
    内容自适应互补频率分解模块 (CA-CFD)
    2D-FFT 物理截断 + 光照流处理
    """
    def __init__(self, dim, prior_channels=31):
        super(CA_OIS, self).__init__()
        self.dynamic_mask = DynamicFrequencyMask(dim)
        self.illumination_stream = PGLA(dim, prior_channels=prior_channels)
        self.reflectance_stream = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, prior_feat):     
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_shifted = torch.fft.fftshift(x_fft)
        
        low_freq_mask = self.dynamic_mask(x)
        high_freq_mask = 1.0 - low_freq_mask
        
        low_freq_fft = x_shifted * low_freq_mask
        high_freq_fft = x_shifted * high_freq_mask
        
        illum_base = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft), norm='ortho').real
        reflect_base = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft), norm='ortho').real
        
        # 光照流进行材质引导注意力平滑
        illum_feat = self.illumination_stream(illum_base, prior_feat)
        # 反射率流进行局部细节提取
        reflect_feat = self.reflectance_stream(reflect_base)
        
        return self.fusion(torch.cat([illum_feat, reflect_feat], dim=1))

# =====================================================================
# Lightweight Spectral-Channel Attention (LSCA)
# =====================================================================

class LSCA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(LSCA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 生成 Q, K, V (1x1 卷积，极速)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        
        # 深度卷积增强局部空间连续性
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        # 生成 QKV 并进行局部空间增强
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑为 [b, head, c/head, h*w]
        # 这里的注意力是在波段（通道）维度上计算的，而不是空间像素上
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        # 归一化，确保点积在合理范围
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 计算通道间的注意力权重矩阵: [b, head, c/head, c/head]
        # 这一步极其轻量，因为 C 很小 (比如 64)，完全规避了空间 N^2 的计算灾难！
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 将注意力权重应用于 V: [b, head, c/head, h*w]
        out = attn @ v
        
        # 恢复空间形状
        out = out.view(b, c, h, w)
        out = self.project_out(out)
        
        # 残差连接，保证初始优化极度平滑
        return x + out

# =====================================================================
# 4. Prior-Guided Adaptive Feature Purification Gate (PG-AFPG)
# =====================================================================
class PG_AFPG(nn.Module):
    """
    物理先验深度引导特征净化门控 (Prior-Guided AFPG)
    """
    def __init__(self, dim, prior_channels=31, init_alpha=0.5):
        super(PG_AFPG, self).__init__()

        self.prior_proj = nn.Sequential(
            nn.Conv2d(prior_channels, dim, kernel_size=1, bias=False),   
            HybridNorm2d(dim, init_alpha),
            nn.GELU()
        )
        self.evaluator = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=False),
            HybridNorm2d(dim, init_alpha),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            HybridNorm2d(dim, init_alpha),
            nn.GELU()
        )

    def forward(self, enc_feat, dec_feat, prior_feat):
        proj_prior = self.prior_proj(prior_feat)
        context = torch.cat([enc_feat, dec_feat, proj_prior], dim=1)
        purification_mask = self.evaluator(context)
        
        purified_enc_feat = enc_feat * purification_mask
        safe_concat = torch.cat([purified_enc_feat, dec_feat], dim=1)
        return self.fusion(safe_concat)

# =====================================================================
# 5. PSRD
# =====================================================================

def make_conv_blocks(dim, num_blocks, residual_scale=0.1):
    return nn.Sequential(
        *[
            ConvBlock(dim, residual_scale=residual_scale)
            for _ in range(num_blocks)
        ]
    )

class ConvBlock(nn.Module):
    def __init__(self, dim, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.residual_scale * self.net(x)

class PSRD(nn.Module):
    def __init__(
        self,
        in_channels=31,
        dim=64,
        num_enc1_blocks=2,
        num_enc2_blocks=2,
        num_bottleneck_blocks=2,
        num_dec_blocks=2,
        num_refine_blocks=1,
        residual_scale=0.1,
        init_alpha=0.5
    ):
        super(PSRD, self).__init__()

        self.nr_ldpe = NR_LDPE()
        
        # 编码器 (1层下采样，兼顾显存与全局感受野)
        self.stem = nn.Conv2d(in_channels, dim, 3, 1, 1)

        self.encoder1 = make_conv_blocks(
            dim,
            num_enc1_blocks,
            residual_scale=residual_scale
        )

        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1)

        self.encoder2 = make_conv_blocks(
            dim * 2,
            num_enc2_blocks,
            residual_scale=residual_scale
        )
        
        # 核心瓶颈层
        self.ca_ois = CA_OIS(dim * 2, prior_channels=in_channels)
        self.bottleneck_attn = LSCA(dim * 2, num_heads=4)

        self.bottleneck_refine = make_conv_blocks(
            dim * 2,
            num_bottleneck_blocks,
            residual_scale=residual_scale
        )

        # 解码器
        self.up1 = nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1)

        self.pgafpg_1 = PG_AFPG(dim=dim, prior_channels=in_channels, init_alpha=init_alpha)

        self.decoder1 = make_conv_blocks(
            dim,
            num_dec_blocks,
            residual_scale=residual_scale
        )

        self.refinement = make_conv_blocks(
            dim,
            num_refine_blocks,
            residual_scale=residual_scale
        )

        self.tail = nn.Conv2d(dim, in_channels, 3, 1, 1)

    def forward(self, x):
        # 1. 物理先验提取与多尺度准备
        prior_feat = self.nr_ldpe(x) 
        prior_feat_half = F.interpolate(prior_feat, scale_factor=0.5, mode='bilinear', align_corners=False)

        # 2. 编码
        feat1 = self.stem(x)
        enc1_out = self.encoder1(feat1)

        feat2 = self.down1(enc1_out)
        enc2_out = self.encoder2(feat2)
        
        # 3. 互补频率分解 (注入对应尺度的材质先验)
        bottle_neck = self.ca_ois(enc2_out, prior_feat_half)
        bottle_neck = self.bottleneck_attn(bottle_neck)
        bottle_neck = self.bottleneck_refine(bottle_neck)
        
        # 4. 解码与物理门控净化
        dec_x = self.up1(bottle_neck)
        dec_x = self.pgafpg_1(enc1_out, dec_x, prior_feat)

        dec_out = self.decoder1(dec_x)
        dec_out = self.refinement(dec_out)
        
        # 5. 全局残差融合 (绝对辐射度锚定)
        out = self.tail(dec_out) + x
        return out

if __name__ == "__main__":
    model = PSRD(
        in_channels=204,
        dim=32,
        num_enc1_blocks=2,
        num_enc2_blocks=2,
        num_bottleneck_blocks=2,
        num_dec_blocks=2,
        num_refine_blocks=1,
        residual_scale=0.1,
        init_alpha=0
        )
    
    # 模拟输入 [Batch, Bands, Height, Width]
    x = torch.rand(1, 204, 256, 256)
    
    print("Forwarding through PSRD architecture...")
    pred = model(x)
    print(f"Output shape: {pred.shape}")

    print("-------------------------------------------")
    from thop import profile, clever_format
    # 使用thop分析模型的运算量和参数量
    MACs, params = profile(model, inputs=(x,),verbose=True)

    # 将结果转换为更易于阅读的格式
    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"FLOPs:{MACs}, Params:{params}")
    print("-------------------------------------------")