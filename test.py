from torchvision.models import mobilenet_v2
from torchsummary import summary

model = mobilenet_v2()
print(model)
summary(model, (3, 224, 224), device = 'cpu')