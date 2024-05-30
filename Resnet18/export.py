'''
    Export Resnet 18 from torchvision Models
    and export as an onnx with image resolition : 1, 3, 224, 224

'''

import torch
import torchvision as tv

model = tv.models.resnet18(weights="DEFAULT")
resnet18_image = torch.rand(1, 3, 224, 224)
torch.onnx.export(model, resnet18_image,  "Udemy_Py_TensorRT/Resnet18/resnet18.onnx")