import segmentation_models_pytorch as smp
from .model_vanilla import UNET

def get_unet_model(type, inchannels, outchannels, weights = None, device = "CUDA"):
	if(type == "vanilla"):
		return UNET(in_features=1, out_features=3).to(device)

	elif(type in ["efficient-b1, resnet34, vgg16, mobilenet-v4"]):
		return smp.Unet(encoder_name=type, encoder_weights=weights,
                         in_channels=1, classes=3).to(device)