import torch.nn as nn
import torch, math
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self):
        super(LabelSmoothingLoss, self).__init__()
    def forward(self, output, label, device = 0):
        C = output.shape[1]
        N = output.shape[0]
        smoothed_labels = torch.full(size=(N,C), fill_value=0.1/(C-1)).to(device)
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(label, dim=1), value=0.9)

        log_prob = torch.nn.functional.log_softmax(output, dim=1)
        return -torch.sum(log_prob * smoothed_labels) / N
class GGL(nn.Module):
    def __init__(self):
        super(GGL, self).__init__()
    def forward(self, batch_feature, batch_label, N):
        # compute similarity
        M = int(batch_feature.shape[0] / N)
        batch_feature = F.normalize(batch_feature)

        mean_feature = torch.mean(batch_feature[0:N], dim = 0)
        mean_feature = mean_feature.unsqueeze(0)
        for i in range(1,M):
            temp_mean_feature = torch.mean(batch_feature[N*i:N*i + N], dim = 0)
            temp_mean_feature = temp_mean_feature.unsqueeze(0)
            mean_feature = torch.cat((mean_feature, temp_mean_feature), dim = 0)

        mean_feature = F.normalize(mean_feature)
        
        # intra loss
        intra_loss = 0
        for i in range(M):
            for j in range(N):
                intra_loss += 2-2*torch.sum(mean_feature[i] * batch_feature[i*N + j].t())
        intra_loss = intra_loss/M/N

        # inter loss
        inter_loss = 0
        for i in range(M):
            for j in range(i+1, M):
                sim = torch.sum(mean_feature[i] * mean_feature[j].t())
                if sim > 0.4:
                    inter_loss += sim
        inter_loss /= M*(M-1)/2

        return intra_loss + inter_loss