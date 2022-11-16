import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


def distMC(Mat_A, Mat_B, norm=1, cpu=False, sq=True):#N by F
    # Mat_A and Mat_B consists of the same normalized embeddings
    N_A = Mat_A.shape[0]
    N_B = Mat_B.shape[0]
    # N x N tensor
    DC = Mat_A.mm(torch.t(Mat_B))
    if cpu:
        if sq:
            DC[torch.eye(N_A).bool()] = -norm
    else:
        if sq:
            # The same embeddings have distance 0 (1 - norm)
            DC[torch.eye(N_A).bool().cuda()] = -norm
            
    return DC

def Mat(Lvec):
    N = Lvec.shape[0]
    # Repeats Lvec N times
    Mask = Lvec.repeat(N, 1)
    # .t() - transpose, we have true in case of 2 labels have the same domain (class)
    Same = (Mask == Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same
    
class EPHNLoss(Module):
    def __init__(self, s=0.1):
        super(EPHNLoss, self).__init__()
        self.semi = False
        self.sigma = s
        
    def forward(self, fvec, Lvec):
        """
        fvec: torch.Tensor
            Batch of encoder result vectors (embeddings)
        Lvec: torch.Tensor
            Batch of ground truth domain for each embedding
        """
        N = Lvec.shape[0]
        # Normalize each vector by its euclidean(p=2) norm
        fvec_norm = F.normalize(fvec, p=2, dim=1)
        # Same domains mask, Not same domains mask
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix (Matrix n x n of scalar product)
        Dist = distMC(fvec_norm, fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = Dist.clone().detach()
        # Where domains are different set to -1
        D_detach_P[Diff] = -1
        # Where the same embedding occured several times set to -1
        D_detach_P[D_detach_P > 0.9999] = -1
        # Values and indexes of maximum similar vector for each other vector in row
        V_pos, I_pos = D_detach_P.max(dim=1)
        # prevent duplicated pairs
        Mask_not_drop_pos = (V_pos > 0)
        # extracting pos similarity score
        Pos = Dist[torch.arange(0, N), I_pos] # flat tensor
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = Dist.clone().detach()
        # Where domains the same set to -1
        D_detach_N[Same] = -1
        if self.semi:
            D_detach_N[(D_detach_N > (V_pos.repeat(N, 1).t())) & Diff] = -1 #extracting SHN
        V_neg, I_neg = D_detach_N.max(dim=1)
        # prevent duplicated pairs
        Mask_not_drop_neg = (V_neg > 0)
        # extracting neg score
        Neg = Dist[torch.arange(0, N), I_neg] # flat tensor
        Neg_log = Neg.clone().detach().cpu()
        
        # triplets
        T = torch.stack([Pos, Neg], 1) # make pairs of pos score and neg score
        Mask_not_drop = Mask_not_drop_pos & Mask_not_drop_neg

        # loss ROW OF SIMILARITY TENSOR IS ANCHOR, POS IS f(Xa) * f(Xep), NEG IS f(Xa) * f(Xhn)
        # We need positive log_softmax score so take the index 0 of columns
        Prob = -F.log_softmax(T / self.sigma, dim=1)[:, 0]
        loss = Prob[Mask_not_drop].mean()

        # print('loss:{:.3f} rt:{:.3f}'.format(loss.item(), Mask_not_drop.float().mean().item()), end='\r')

        easy_positive_scores = torch.cat([Pos_log[(V_pos > 0)], torch.Tensor([0, 1])], dim=0)
        hard_negative_scores = torch.cat([Neg_log[(V_neg > 0)], torch.Tensor([0,1])], dim=0)
        # Triplet loss margin
        pos_neg_difference = Pos_log.mean() - Neg_log.mean()

        return loss, easy_positive_scores, hard_negative_scores, pos_neg_difference
    