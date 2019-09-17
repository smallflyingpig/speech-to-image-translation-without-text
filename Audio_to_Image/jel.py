import torch
import torch.nn as nn 

class JointEmbeddingLossLayer(nn.Module):
    def __init__(self, loss_diff_coeff, loss_same_coeff):
        super(JointEmbeddingLossLayer, self).__init__()
        self.loss_diff_coeff = loss_diff_coeff
        self.loss_same_coeff = loss_same_coeff
    
    def forward(self, fea_txt, fea_img, label):
        loss, grad, accu = JointEmbeddingLoss(
            fea_txt, fea_img, label, self.loss_diff_coeff, self.loss_same_coeff
            )
        return {'loss':loss, 'grad':grad, 'accu':accu}

        
def JointEmbeddingLoss(fea_txt, fea_img, label, loss_diff_coeff, loss_same_coeff):
    batchsize = fea_img.size(0)
    num_class = fea_txt.size(0)
    score = torch.zeros(batchsize, num_class)
    txt_grads = torch.zeros_like(fea_txt, requires_grad=True)
    img_grads = torch.zeros_like(fea_img, requires_grad=True)

    #score = torch.mm(fea_txt.cpu(), fea_img.cpu().transpose(0,1))
    score = torch.mm(fea_img, fea_txt.transpose(0,1))

    score_abs = score - score.diag()
    selected_idx_diff = (label.unsqueeze(0).repeat([batchsize, 1])!=label.unsqueeze(1))
    selected_idx_same = (label.unsqueeze(0).repeat([batchsize, 1])==label.unsqueeze(1))
    loss_diff = score_abs[selected_idx_diff]+1
    loss_same = score_abs[selected_idx_same]
    # print(score.detach().tolist())

    loss = (loss_diff_coeff * loss_diff[loss_diff>0].sum()+ loss_same_coeff * loss_same[loss_same>0].sum())
    # loss = loss_diff[loss_diff>0].sum()

    _, max_idx = score.max(dim=1)
    acc_batch = (max_idx==torch.LongTensor(range(score.shape[1])).to(max_idx.device)).sum().cpu().item()

    acc_batch = 100 * (acc_batch / batchsize)
    denom = batchsize * num_class

    return loss/denom, [txt_grads/denom, img_grads/denom], acc_batch
