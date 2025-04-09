import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
	def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
		super(MLP, self).__init__()
		self.fcs = nn.ModuleList()
		self.hidden_sizes = hidden_sizes
		total_fc_sizes = [input_size, *self.hidden_sizes, output_size]
		for i in range(len(total_fc_sizes) - 1):
			self.fcs.append(nn.Linear(total_fc_sizes[i], total_fc_sizes[i + 1]))

	def forward(self, x):
		for i, fc in enumerate(self.fcs):
			x = fc(x)
			if i < len(self.fcs) - 1:
				x = F.relu(x)
		return x


class TimeMLP(nn.Module):
	def __init__(self, input_size: int, hidden_sizes: list, output_size: int, time_dim: int):
		super(TimeMLP, self).__init__()
		self.hidden_sizes = hidden_sizes
		self.time_dim = time_dim

		# 使用正弦位置编码替代 Linear 作为时间步编码
		self.time_dim = time_dim
		self.time_embedding = self.sinusoidal_embedding

		# MLP 网络，输入变为 input_size + time_dim
		total_fc_sizes = [input_size + time_dim, *self.hidden_sizes, output_size]
		self.fcs = nn.ModuleList([
			nn.Linear(total_fc_sizes[i], total_fc_sizes[i + 1])
			for i in range(len(total_fc_sizes) - 1)
		])

	def sinusoidal_embedding(self, t):
		"""
		生成 Sinusoidal 时间步编码
		:param t: shape = [batch_size, 1] 或 [batch_size]
		:return: shape = [batch_size, time_dim]
		"""
		if t.dim() == 1:
			t = t.unsqueeze(-1)  # 确保 t 变成 [batch_size, 1]

		batch_size, _ = t.shape
		half_dim = self.time_dim // 2
		emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) *
						(-math.log(10000.0) / (half_dim - 1)))
		emb = t * emb  # shape: [batch_size, half_dim]

		# 拼接 sin 和 cos
		sin_emb = torch.sin(emb)
		cos_emb = torch.cos(emb)
		time_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # shape: [batch_size, time_dim]

		return time_emb

	def forward(self, x, t):
		"""
		:param x: 输入数据, shape: [batch_size, input_size]
		:param t: 时间步, shape: [batch_size] or [batch_size, 1]
		"""
		# 计算时间步的正弦位置编码
		t_embedding = self.time_embedding(t)  # shape: [batch_size, time_dim]

		# 拼接时间信息到输入数据
		x = torch.cat([x, t_embedding], dim=-1)  # shape: [batch_size, input_size + time_dim]

		# 通过 MLP 计算
		for i, fc in enumerate(self.fcs):
			x = fc(x)
			if i < len(self.fcs) - 1:
				x = F.silu(x)  # 适合扩散模型等任务
		return x
