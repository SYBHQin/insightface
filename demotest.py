import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

print(torch.cuda.get_device_name(0))

x = torch.randn(3,3).cuda()
print(x)