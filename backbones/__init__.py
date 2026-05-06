#from .mobilefacenet import get_mbf_large
import torch
import torch.nn as nn
# import timm
import backbones.mobilefacenet
#import backbones.mobilefacenext
#import backbones.mobilefacenet_huge
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
# from efficientnet_pytorch import EfficientNet
from typing import Callable

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def replace_swish_with_onnx_swish(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            if module.__class__.__name__ == 'MemoryEfficientSwish':
                setattr(model, name, Swish())
            else:
                replace_swish_with_onnx_swish(module)

def get_model(args, logger, **kwargs):
    
    fp16 = True
    num_features = 512

    if args.model_name  == "mbf":
        num_features = 256
        net = backbones.mobilefacenet.get_mbf(fp16=fp16, num_features=num_features)

    elif args.model_name  == "mbf_large":
        net = backbones.mobilefacenet.get_mbf_large(fp16=fp16, num_features=num_features)
        
    elif args.model_name == 'mbf_huge':
        net = backbones.mobilefacenet_huge.get_mbf_huge(fp16=fp16, num_features=num_features)

        
    # elif args.model_name == 'resnet14t':
    #     num_features = 2048
    #     net = timm.create_model('resnet14t', pretrained=False, checkpoint_path='/data4022/shayouyang/resnet_weights/timm_resnet14t.pth')
    #     net.fc = nn.Sequential()
        
    elif args.model_name == 'iresnet50':
        num_features = 512
        net = iresnet50(pretrained=False, fp16=True, num_features=num_features, **kwargs)
        
            
    # elif args.model_name == 'efficientnet-b0':
    #     num_features = 1280
    #     net = EfficientNet.from_pretrained('efficientnet-b0')

    #     net._fc = nn.Sequential()
    #     net._dropout = nn.Sequential()

    #     replace_swish_with_onnx_swish(net)


    if logger: logger.info('Loading {} models'.format(args.model_name))
    return net, num_features