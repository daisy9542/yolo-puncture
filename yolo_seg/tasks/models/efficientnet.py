
from efficientnet_pytorch import EfficientNet
from timm.models.registry import register_model

@register_model
def efficientnet_b3(pretrained=True, num_classes=2, **kwargs):
    model_name = 'efficientnet-b3'
    if pretrained:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else: 
        model = EfficientNet.from_name(model_name, num_classes=num_classes)
    return model


@register_model
def efficientnet_b4(pretrained=True, num_classes=2, **kwargs):
    model_name = 'efficientnet-b4'
    if pretrained:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else: 
        model = EfficientNet.from_name(model_name, num_classes=num_classes)
    return model


@register_model
def efficientnet_b5(pretrained=True, num_classes=2, **kwargs):
    model_name = 'efficientnet-b5'
    if pretrained:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else: 
        model = EfficientNet.from_name(model_name, num_classes=num_classes)
    return model


@register_model
def efficientnet_b7(pretrained=True, num_classes=2, **kwargs):
    model_name = 'efficientnet-b7'
    if pretrained:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    else: 
        model = EfficientNet.from_name(model_name, num_classes=num_classes)
    return model
