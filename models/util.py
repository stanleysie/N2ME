import torch

def GANLoss(predicted, is_real):
    return -torch.mean(predicted) if is_real else torch.mean(predicted)

def TVLoss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = torch.numel(x[:,:,1:,:])
    count_w = torch.numel(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:] - x[:,:, :h_x -1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:, :w_x -1]),2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size