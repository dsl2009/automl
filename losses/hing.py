import torch
def hing_loss(pred,target,delta=1.0):
    #pred = torch.sigmoid(pred)
    B = pred.size(0)
    ix = torch.arange(0, B).cuda().long()
    sc = pred[ix, target]
    lbs = sc.view(-1, 1).expand_as(pred)
    loss = pred - lbs + delta
    loss = torch.clamp(loss, min=0)
    loss[ix, target] = 0.0
    return loss.mean()*B
