import numpy
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ConvTwice(nn.Module):

	def __init__(self, in_features, out_features):
		super(ConvTwice, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(in_features, out_features, kernel_size = (3,3), stride = 1, padding = 1, bias = False),
			nn.BatchNorm2d(out_features),
			nn.ReLU(inplace = True),

			nn.Conv2d(out_features, out_features, kernel_size = (3,3), stride = 1, padding = 1, bias = False),
			nn.BatchNorm2d(out_features),
			nn.ReLU(inplace = True),
		)

	def forward(self, x):

		return self.conv(x)



class UNET(nn.Module):

	def __init__(self, in_features, out_features, feature_list = [64, 128, 256, 512]):
		super(UNET, self).__init__()
		self.upSampling = nn.ModuleList()
		self.downSampling = nn.ModuleList()

		self.maxpool = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
		self.BN = ConvTwice(feature_list[-1], feature_list[-1] * 2)

		for feature in feature_list:
			self.downSampling.append( ConvTwice(in_features, feature) )
			in_features = feature


		for feature in reversed(feature_list):
			self.upSampling.append( nn.ConvTranspose2d(feature * 2, feature, kernel_size = (2, 2), stride = 2) )
			self.upSampling.append( ConvTwice(feature * 2, feature) )

		self.output = nn.Conv2d(feature_list[0], out_features, kernel_size = (1, 1))


	def forward(self, x):

		skip_connections = []

		for down in self.downSampling:

			x = down(x)
			skip_connections.append(x)
			x = self.maxpool(x)


		x = self.BN(x)

		skip_connections = skip_connections[::-1]

		for index in range(0, len(self.upSampling), 2):

			x = self.upSampling[index](x)
			if x.shape != skip_connections[index // 2].shape:

				x = TF.resize(x, size=skip_connections[index // 2].shape[2:])

			cat = torch.cat([skip_connections[index // 2], x], dim = 1)

			x = self.upSampling[index + 1](cat)

		return self.output(x)