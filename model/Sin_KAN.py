import torch
import torch.nn.functional as F
import math


class SPKAL(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,  # 网格大小
            spline_order=3,  # B样条  基函数的阶次
            scale_noise=0.1,  # 初始化 spline 曲线时引入的小扰动量
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,  # 基础线性部分的激活函数，默认是 SiL
            grid_eps=0.02,
            grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(SPKAL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 分段多项式的阶数
        self.num_basis_per_feature = self.grid_size
        h = (grid_range[1] - grid_range[0]) / grid_size  # 计算网格步长
        # torch.arange(起点，终点） 默认步长是1 使用个*h对生成的序列进行缩放
        xx = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]  # 生成网格
        grid = (
            (xx)
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # 初始化基础权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order + spline_order)
        )  # 初始化分段多项式权重  可学习的控制点

        # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (  # 生成缩放噪声
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )

            # 计算分段多项式权重
            a4 = self.curve2coeff(
                self.grid.T[self.spline_order: -self.spline_order],
                noise,
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * a4
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def trig_basis(self, x: torch.Tensor):
        """
        构造基于三角函数的分段局部基函数 (正弦窗)
        Args:
            x: (B, in_features)
        Returns:
            basis: (B, in_features, grid_size)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        x = x.unsqueeze(-1)  # [B, in_features, 1]

        a = self.grid[:, :-1]  # [in_features, grid_size]
        b = self.grid[:, 1:]  # [in_features, grid_size]
        interval = b - a  # [in_features, grid_size]

        # Expand a, b for broadcasting
        a = a.unsqueeze(0)  # [1, in_features, grid_size]
        b = b.unsqueeze(0)  # [1, in_features, grid_size]
        interval = interval.unsqueeze(0)  # [1, in_features, grid_size]

        # 局部正弦窗函数
        inside_mask = ((x >= a) & (x < b)).to(x.dtype)  # [B, in_features, grid_size]
        normed_x = (x - a) / interval.clamp(min=1e-8)  # Normalize to [0, 1]
        trig_val = torch.sin(math.pi * normed_x)  # 正弦窗 [0, pi] → [0, 1]

        basis = inside_mask * trig_val  # Apply mask
        return basis  # shape: (B, in_features, grid_size)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.trig_basis(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order + self.spline_order,
        )
        return result.contiguous()  # (out_features, in_features, grid_size + spline_order)

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """

        z = (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)
        return self.spline_weight * (z)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        ENM_result = F.linear(self.base_activation(x), self.base_weight)

        I_Non = self.trig_basis(x)
        I_Non_v = I_Non.contiguous().view(I_Non.size(0), -1)

        learnable_weight = self.scaled_spline_weight.view(self.out_features, -1)

        SNM_result = F.linear(I_Non_v, learnable_weight, )  #

        return ENM_result + SNM_result
