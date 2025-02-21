import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.utils.misc import concat_all_gather_no_grad


class PriorKnowledgeGuider():
    def __init__(self, arch, clip, pkg_labels, debiased_labelers, confidence_mask=False, marginal_loss=False, reduction='mean'):
        super(PriorKnowledgeGuider, self).__init__()
        self.arch = arch
        self.clip = clip
        self.pkg_labels = pkg_labels
        self.confidence_mask = confidence_mask
        self.marginal_loss = marginal_loss
        self.reduction = reduction

        self.debiased_labelers = debiased_labelers
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        if self.arch == 'BiomedCLIP':
            self.image_encoder = clip.visual
            self.text_encoder = clip.text
            self.logit_scale = clip.logit_scale.exp().detach()
            self.image_encoder.eval()
            self.text_encoder.eval()
            
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        elif self.arch == 'CLIP':
            self.logit_scale = clip.logit_scale.exp().detach()
            self.clip.eval()
            for param in self.clip.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode_imgs(self, imgs, normalize=True):
        imgs = self.clip.encode_image(imgs) if self.arch == 'CLIP' else self.clip.image_encoder(imgs)
        imgs = F.normalize(imgs, dim=-1) if normalize else imgs
        return imgs

    @torch.no_grad()
    def cache_texts(self, tokenizer, device, context_length=256, normalize=True):
        self.texts = {}
        if self.arch == 'BiomedCLIP':
            for label_type, labels in self.pkg_labels.items():
                tokenized = tokenizer(labels, context_length=context_length).to(device)
                text_features = self.text_encoder(tokenized).to(device)
                self.texts[label_type] = F.normalize(text_features, dim=-1) if normalize else text_features
        elif self.arch == 'CLIP':
            for label_type, labels in self.pkg_labels.items():
                tokenized = tokenizer(labels).to(device)
                text_features = self.clip.encode_text(tokenized).to(device)
                self.texts[label_type] = F.normalize(text_features, dim=-1) if normalize else text_features
                # self.texts[label_type] = F.normalize(torch.randn_like(text_features), dim=-1) if normalize else text_features

    def compute_loss(self, image_features, target_features):
        loss_collection = {}
        acc_collection = {}
        for label_type, text_features in self.texts.items():
            debiased_labeler = self.debiased_labelers[label_type]
            with torch.no_grad():
                target_logits = self.logit_scale * target_features @ text_features.T
                target_labels, mask = debiased_labeler.debiased_labeling(target_logits)
                debiased_labeler.update_counterfactual(target_logits)

            image_logits = self.logit_scale * image_features @ text_features.T
            if self.marginal_loss:
                image_logits = image_logits + debiased_labeler.get_margin()

            loss = self.ce_loss(image_logits, target_labels)            
            if self.confidence_mask:
                loss = loss * mask
                loss_mean = loss.sum() / max(mask.sum(), 1)
                acc = (image_logits.argmax(dim=-1) == target_labels).float() * mask
                acc_mean = acc.sum() / max(mask.sum(), 1)
            else:
                loss_mean = loss.mean()
                acc = (image_logits.argmax(dim=-1) == target_labels).float()
                acc_mean = acc.mean()
            loss_collection[label_type] = loss_mean
            acc_collection[label_type] = acc_mean

        return self.reduce(loss_collection), acc_collection

    def reduce(self, loss_collection):
        if self.reduction == 'sum':
            return torch.stack(list(loss_collection.values())).sum()
        elif self.reduction == 'mean':
            return torch.stack(list(loss_collection.values())).mean()
        elif self.reduction == 'none':
            return loss_collection
        else:
            raise NotImplementedError('Inapplicable reduction method.')


class DebiasedPL(nn.Module):
    def __init__(self, num_classes, debias_factor, momentum, threshold, masked_counterfactual=False):
        super(DebiasedPL, self).__init__()
        self.num_classes = num_classes
        self.debias_factor = debias_factor
        self.momentum = momentum
        self.threshold = threshold
        self.masked_counterfactual = masked_counterfactual
        
        self.counterfactual = None
        
    def init_counterfactual(self, device):
        self.counterfactual = (torch.ones([1, self.num_classes], dtype=torch.float) / self.num_classes).to(device)

    def causal_inference(self, logits):
        debiased_prob = F.softmax(logits - self.debias_factor * torch.log(self.counterfactual), dim=-1)
        return debiased_prob
    
    def update_counterfactual(self, logits):
        logits = logits.detach()
        logits = concat_all_gather_no_grad(logits)
        probs = F.softmax(logits, dim=-1)
        if self.masked_counterfactual:
            mask = (probs.max(dim=-1).values > self.threshold).float()
            mean_prob = probs * mask.unsqueeze(-1)
            mean_prob = mean_prob.sum(dim=0) / mask.sum()
        else:
            mean_prob = probs.mean(dim=0)
        self.counterfactual = self.momentum * self.counterfactual + (1 - self.momentum) * mean_prob

    def debiased_labeling(self, logits):
        debiased_prob = self.causal_inference(logits)
        debiased_label = debiased_prob.argmax(dim=-1)
        # mask = (debiased_prob.max(dim=-1).values > self.threshold).float()
        for i in range(self.num_classes):
            mask = (debiased_label == i).float()
            pred_prob = debiased_prob[:, i]
            topk_threshold = torch.kthvalue(pred_prob, int(len(pred_prob) * 0.8)).values
            mask = mask * (pred_prob >= topk_threshold).float()
        
        return debiased_label, mask

    def get_margin(self):
        return self.debias_factor * torch.log(self.counterfactual)
