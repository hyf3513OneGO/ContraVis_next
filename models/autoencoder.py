import torch
from torch import nn
import torch.nn.functional as F
class Autoencoder(nn.Module):
	def __init__(self, encoder, decoder):
		super(Autoencoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, x):
		latent = self.encoder(x)
		output = self.decoder(latent)
		mse = F.mse_loss(output, x)
		return output,latent,mse

	def encode(self, x):
		self.encoder.train()
		return self.encoder(x)

	def decode(self, latent):
		self.decoder.train()
		return self.decoder(latent)

	def encode_no_gradient(self, x):
		self.encoder.eval()
		with torch.no_grad():
			return self.encoder(x)

	def decode_no_gradient(self, latent):
		self.decoder.eval()
		with torch.no_grad():
			return self.decoder(latent)
