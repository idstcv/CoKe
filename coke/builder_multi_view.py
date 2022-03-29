# Copyright (c) Alibaba Group
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoKe(nn.Module):
    """
    Build a CoKe model with multiple clustering heads
    """

    def __init__(self, base_encoder, K, dim=128, num_ins=1281167, num_head=3, T=0.05, dual_lr=20., ratio=0.4):
        super(CoKe, self).__init__()
        self.T = T
        self.K = K
        self.dual_lr = dual_lr
        self.ratio = ratio
        self.lb = [ratio / k for k in self.K]
        self.num_head = num_head
        # create the encoder with projection head
        self.encoder = base_encoder(num_classes=dim)

        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                        nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim), nn.BatchNorm1d(dim))
        # prediction head
        self.predictor = nn.Sequential(nn.Linear(dim, dim_mlp), nn.BatchNorm1d(dim_mlp),
                                       nn.ReLU(inplace=True), nn.Linear(dim_mlp, dim))
        # decoupled cluster assignments
        self.register_buffer("assign_labels", torch.ones(num_head, num_ins, dtype=torch.long))
        self.pre_centers = []
        self.cur_centers = []
        self.duals = []
        self.counters = []
        for i in range(0, self.num_head):
            centers = F.normalize(torch.randn(dim, self.K[i]), dim=0)
            self.register_buffer("pre_center_" + str(i), centers.clone())
            self.register_buffer("cur_center_" + str(i), centers.clone())
            self.register_buffer("dual_" + str(i), torch.zeros(self.K[i]))
            self.register_buffer("counter_" + str(i), torch.zeros(self.K[i]))

    @torch.no_grad()
    def load_param(self):
        for i in range(0, self.num_head):
            self.pre_centers.append(getattr(self, "pre_center_" + str(i)))
            self.cur_centers.append(getattr(self, "cur_center_" + str(i)))
            self.duals.append(getattr(self, "dual_" + str(i)))
            self.counters.append(getattr(self, "counter_" + str(i)))

    @torch.no_grad()
    def gen_label(self, feats, branch):
        return torch.argmax(torch.mm(feats, self.cur_centers[branch]) + self.duals[branch], dim=1).squeeze(-1)

    @torch.no_grad()
    def update_label(self, targets, labels, branch):
        self.assign_labels[branch][targets] = labels

    @torch.no_grad()
    def get_label(self, target, branch):
        return self.assign_labels[branch][target]

    @torch.no_grad()
    def update_center(self):
        for i in range(0, self.num_head):
            self.pre_centers[i] += self.cur_centers[i].clone() - self.pre_centers[i]
            self.counters[i] = torch.zeros(self.K[i]).cuda()

    @torch.no_grad()
    def update_center_mini_batch(self, feats, labels, branch):
        label_idx, label_count = torch.unique(labels, return_counts=True)
        self.duals[branch][label_idx] -= self.dual_lr / len(labels) * label_count
        self.duals[branch] += self.dual_lr * self.lb[branch]
        if self.ratio < 1:
            self.duals[branch][self.duals[branch] < 0] = 0
        alpha = self.counters[branch][label_idx].float()
        self.counters[branch][label_idx] += label_count
        alpha = alpha / self.counters[branch][label_idx].float()
        self.cur_centers[branch][:, label_idx] = self.cur_centers[branch][:, label_idx] * alpha
        self.cur_centers[branch].index_add_(1, labels, feats.data.T * (1. / self.counters[branch][labels]))
        self.cur_centers[branch][:, label_idx] = F.normalize(self.cur_centers[branch][:, label_idx], dim=0)

    def forward(self, view1, view2, sview, target, epoch):
        x1 = self.encoder(view1)
        x1_pred = self.predictor(x1)
        x1_pred = F.normalize(x1_pred, dim=1)
        x1_proj = F.normalize(x1, dim=1)
        x2 = self.encoder(view2)
        x2_pred = self.predictor(x2)
        x2_pred = F.normalize(x2_pred, dim=1)
        x2_proj = F.normalize(x2, dim=1)

        pred_small = []
        proj_small = []
        pred_view1 = []
        pred_view2 = []
        proj_view1 = []
        proj_view2 = []
        sx_proj = []
        sx_pred = []
        snum = len(sview)
        for i in range(0, snum):
            sx = self.encoder(sview[i])
            sx_p = self.predictor(sx)
            sx_pred.append(F.normalize(sx_p, dim=1))
            sx_proj.append(F.normalize(sx, dim=1))

        for i in range(0, self.num_head):
            proj_view1.append(x1_proj.matmul(self.pre_centers[i]) / self.T)
            proj_view2.append(x2_proj.matmul(self.pre_centers[i]) / self.T)
            pred_view1.append(x1_pred.matmul(self.pre_centers[i]) / self.T)
            pred_view2.append(x2_pred.matmul(self.pre_centers[i]) / self.T)
            for j in range(0, snum):
                proj_small.append(sx_proj[j].matmul(self.pre_centers[i]) / self.T)
                pred_small.append(sx_pred[j].matmul(self.pre_centers[i]) / self.T)

        with torch.no_grad():
            feat_mean = (x1_proj + x2_proj) * 0.5
            targets = concat_all_gather(target)
            feats = concat_all_gather(feat_mean)
            cur_labels = []
            if epoch == 0:
                for j in range(0, self.num_head):
                    labels = self.gen_label(feats, j)
                    self.update_label(targets, labels, j)
                    self.update_center_mini_batch(feats, labels, j)
                    cur_labels.append(self.get_label(target, j))
            else:
                for j in range(0, self.num_head):
                    # obtain labels from last epoch
                    cur_labels.append(self.get_label(target, j))
                    labels = self.gen_label(feats, j)
                    self.update_center_mini_batch(feats, labels, j)
                    self.update_label(targets, labels, j)
        return pred_view1, pred_view2, proj_view1, proj_view2, pred_small, proj_small, cur_labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
