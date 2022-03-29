# Copyright (c) Alibaba Group
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoKe(nn.Module):
    """
    Build a CoKe model with multiple clustering heads
    """

    def __init__(self, base_encoder, K, dim=128, num_ins=1281167, num_head=3, T=0.1, dual_lr=20, stage=801, t=0.5,
                 ratio=0.4, ls=5):
        super(CoKe, self).__init__()
        self.T = T
        self.K = K
        self.dual_lr = dual_lr
        self.ratio = ratio
        self.lb = [ratio / k for k in self.K]
        self.dual_lr = dual_lr
        self.ls = ls  # non-zero label size in second stage
        self.stage = stage  # number of epochs for the first stage
        self.t = t  # temperature for label smoothing
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
        self.register_buffer("assign_labels", torch.ones(num_head, num_ins, ls, dtype=torch.long))
        self.register_buffer("label_val", torch.zeros(num_head, num_ins, ls))
        self.register_buffer("label_idx", torch.zeros(num_head, num_ins, dtype=torch.long))

    @torch.no_grad()
    def load_param(self):
        for i in range(0, self.num_head):
            self.pre_centers.append(getattr(self, "pre_center_" + str(i)))
            self.cur_centers.append(getattr(self, "cur_center_" + str(i)))
            self.duals.append(getattr(self, "dual_" + str(i)))
            self.counters.append(getattr(self, "counter_" + str(i)))

    @torch.no_grad()
    def gen_label(self, feats, epoch, branch):
        if epoch >= self.stage:
            return torch.argmax(torch.mm(feats, self.pre_centers[branch]) + self.duals[branch], dim=1).squeeze(-1)
        else:
            return torch.argmax(torch.mm(feats, self.cur_centers[branch]) + self.duals[branch], dim=1).squeeze(-1)

    @torch.no_grad()
    def update_label(self, targets, labels, epoch, branch):
        if epoch < self.stage or self.ls == 1:
            self.assign_labels[branch][targets, 0] = labels
        else:
            if epoch == self.stage:
                self.assign_labels[branch][targets, 0] = labels
                self.label_val[branch][targets, 0] = 1.
                self.label_idx[branch][targets] = 1
            else:
                factor = 1. / (epoch - self.stage + 1.)
                tmp = (self.assign_labels[branch][targets, :] - labels.reshape(-1, 1) == 0).nonzero(as_tuple=False)
                idx = self.label_idx[branch][targets]
                val = self.label_val[branch][targets, idx]
                if len(tmp[:, 0]) > 0:
                    idx[tmp[:, 0]] = tmp[:, 1]
                    val[tmp[:, 0]] = 0.
                self.label_val[branch][targets, idx] -= val
                self.label_val[branch][targets, :] *= (1. - factor) / (1. - val.reshape(-1, 1))
                self.assign_labels[branch][targets, idx] = labels
                self.label_val[branch][targets, idx] += factor
                self.label_idx[branch][targets] = torch.min(self.label_val[branch][targets, :], dim=1).indices

    @torch.no_grad()
    def get_label(self, target, epoch, branch):
        if epoch <= self.stage or self.ls == 1:
            return self.assign_labels[branch][target, 0]
        else:
            labels = torch.zeros(len(target), self.K[branch]).cuda()
            for i, t in enumerate(target):
                labels[i, :].index_add_(0, self.assign_labels[branch][t.item(), :], self.label_val[branch][t.item(), :])
            labels[labels > 0] = torch.exp(labels[labels > 0] / self.t)
            labels /= torch.sum(labels, dim=1, keepdim=True)
            return labels

    @torch.no_grad()
    def update_center(self, epoch):
        if epoch < self.stage:
            for i in range(0, self.num_head):
                self.pre_centers[i] += self.cur_centers[i].clone() - self.pre_centers[i]
        if epoch >= self.stage:
            factor = 1. / (epoch - self.stage + 1.)
            for i in range(0, self.num_head):
                tmp_center = F.normalize(self.cur_centers[i], dim=0)
                self.pre_centers[i] += F.normalize((1. - factor) * self.pre_centers[i] + factor * tmp_center, dim=0) - \
                                       self.pre_centers[i]
                self.cur_centers[i] += self.pre_centers[i].clone() - self.cur_centers[i]
        for i in range(0, self.num_head):
            self.counters[i] = torch.zeros(self.K[i]).cuda()

    @torch.no_grad()
    def update_center_mini_batch(self, feats, labels, epoch, branch):
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
        if epoch < self.stage:
            self.cur_centers[branch][:, label_idx] = F.normalize(self.cur_centers[branch][:, label_idx], dim=0)

    def forward(self, img, target, epoch):
        x = self.encoder(img)
        x_pred = self.predictor(x)
        x_pred = F.normalize(x_pred, dim=1)
        x_proj = F.normalize(x, dim=1)

        pred_view = []
        proj_view = []

        for i in range(0, self.num_head):
            proj_view.append(x_proj.matmul(self.pre_centers[i]) / self.T)
            pred_view.append(x_pred.matmul(self.pre_centers[i]) / self.T)

        with torch.no_grad():
            targets = concat_all_gather(target)
            feats = concat_all_gather(x_proj)
            cur_labels = []
            if epoch == 0:
                for j in range(0, self.num_head):
                    labels = self.gen_label(feats, epoch, j)
                    self.update_label(targets, labels, epoch, j)
                    self.update_center_mini_batch(feats, labels, epoch, j)
                    cur_labels.append(self.get_label(target, epoch, j))
            else:
                for j in range(0, self.num_head):
                    cur_labels.append(self.get_label(target, epoch, j))
                    labels = self.gen_label(feats, epoch, j)
                    self.update_center_mini_batch(feats, labels, epoch, j)
                    self.update_label(targets, labels, epoch, j)
        return pred_view, proj_view, cur_labels


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
