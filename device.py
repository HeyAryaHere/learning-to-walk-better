import torch


device = torch.device('cuda:0') 
torch.cuda.empty_cache()
print("Device set to : " + str(torch.cuda.get_device_name(device)))
