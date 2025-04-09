import torch
import torch.nn.functional as F


class RectifiedFlow:
	@staticmethod
	def linear_interpolate(x_1, t, x_0=None):
		if x_0 is None:
			x_0 = torch.randn_like(x_1)
		t = t.reshape(t.shape + (1,) * (len(x_1.shape)-1))
		x_t = (1 - t) * x_0 + t * x_1
		return x_t

	@staticmethod
	def mse_loss(v, x_1, x_0):
		loss = F.mse_loss(v, x_1 - x_0)
		return loss

	@staticmethod
	def euler_step_forward(x_t, v, dt):
		x_t = x_t + v * dt
		return x_t

	@staticmethod
	def euler_step_backward(x_t, v, dt):
		x_t = x_t - v * dt
		return x_t


def fm_infer(pred_model, x_0, infer_type="forward", steps=10, device="cuda"):
	device = torch.device(device)
	pred_model = pred_model.to(device)
	pred_model.eval()
	rf = RectifiedFlow()
	with torch.no_grad():
		x_t = x_0.clone().to(device)
		dt = 1 / steps
		for j in range(steps):
			t = j * dt
			t_tensor = t*torch.ones(x_0.shape[0]).unsqueeze(-1).to(device)
			if infer_type == "forward":
				v_pred = pred_model(x_t,t_tensor)
				x_t = rf.euler_step_forward(x_t, v_pred, dt)
			elif infer_type == "backward":
				v_pred = pred_model(x_t,t_tensor)
				x_t = rf.euler_step_backward(x_t, v_pred, dt)
	return x_t
