# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
from abc import get_cache_token
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math
import numpy as np

class PrototypePLoss(nn.Module):
    def __init__(self, num_classes, temperature):
        super(PrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.arange(num_classes)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes)

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.not_equal(labels, labels.T)
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C+N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), labels)
        return loss
    
class MultiDomainPrototypePLoss(nn.Module):
    def __init__(self, num_classes, num_domains, temperature):
        super(MultiDomainPrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.num_classes = num_classes
        self.label = torch.arange(num_classes)
        self.domain_label = torch.arange(num_domains)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels, domain_labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes.reshape(-1, prototypes.size(-1)))

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.logical_or(torch.not_equal(labels, labels.T), torch.not_equal(domain_labels, domain_labels))
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C*D + N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), domain_labels * self.num_classes + labels)
        return loss

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
