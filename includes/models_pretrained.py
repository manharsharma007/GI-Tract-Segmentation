import segmentation_models_pytorch as smp
from .model_vanilla import UNET

def get_unet_model(type_, inchannels, outchannels, weights = None, device = "cuda"):
	if(type_ == "vanilla"):
		return UNET(in_features=1, out_features=3).to(device)

	elif(type_ in ["efficientnet-b1", "resnet34", "vgg16", "mobilenet_v2", "resnet50", "efficientnet-b3"]):
		return smp.Unet(encoder_name=type_, encoder_weights=weights,
                         in_channels=1, classes=3).to(device)