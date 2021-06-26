import torch
x = torch.load('./weights.pth')
x["fc1.0.weight"] = x["fc.0.weight"]
x["fc1.0.bias"] = x["fc.0.bias"]
x["fc2.0.weight"] = x["fc.2.weight"]
x["fc2.0.bias"] = x["fc.2.bias"]

del x["fc.0.bias"]
del x["fc.0.weight"]
del x["fc.2.bias"]
del x["fc.2.weight"]

torch.save(x, './weights.pth')
