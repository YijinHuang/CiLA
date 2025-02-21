import os
os.environ['CURL_CA_BUNDLE'] = ''
import time
import shutil
import random

import hydra
import torch
import open_clip
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
from solo.methods.base import BaseMethod
from solo.args.linear import parse_cfg


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    print(cfg.finetuning)

    save_path = cfg.finetuning.save_path
    log_path = cfg.finetuning.log_path
    if log_path is None:
        log_path = os.path.join(save_path, 'log')
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(log_path)

    set_random_seed(cfg.finetuning.seed)
    model = generate_model_from_solo_learn(cfg)
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = Estimator(cfg.finetuning.criterion, cfg.finetuning.num_classes)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)

    shutil.rmtree(save_path)


def train(cfg, model, train_dataset, val_dataset, estimator, logger=None, scaler=None):
    device = cfg.finetuning.device
    optimizer = torch.optim.Adam(
        model.head.parameters() if cfg.finetuning.linear else model.parameters(),
        lr=cfg.finetuning.learning_rate,
        weight_decay=cfg.finetuning.weight_decay
    )
    loss_function = nn.MSELoss() if cfg.finetuning.criterion == 'mse' else nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.finetuning.epochs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.finetuning.batch_size,
        shuffle=True,
        num_workers=cfg.finetuning.num_workers,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.finetuning.batch_size,
        num_workers=cfg.finetuning.num_workers,
        pin_memory=True
    )

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(cfg.finetuning.epochs):
        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if not cfg.finetuning.disable_progress else enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device).long()

            y_pred = model(X)
            y_pred = y_pred.squeeze() if cfg.finetuning.criterion == 'mse' else y_pred
            loss = loss_function(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc = estimator.get_accuracy(6)
            avg_kappa = estimator.get_kappa(6)

            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            message = '[{}] epoch: [{} / {}], loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}'.format(current_time, epoch + 1, cfg.finetuning.epochs, avg_loss, avg_acc, avg_kappa)
            if not cfg.finetuning.disable_progress:
                progress.set_description(message)

        if cfg.finetuning.disable_progress:
            print(message)

        # validation performance
        if epoch % cfg.finetuning.eval_interval == 0:
            eval(model, val_loader, estimator, device)
            acc = estimator.get_accuracy(6)
            kappa = estimator.get_kappa(6)
            print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
            if logger:
                logger.add_scalar('validation accuracy', acc, epoch)
                logger.add_scalar('validation kappa', kappa, epoch)

            # save model
            indicator = kappa if cfg.finetuning.kappa_prior else acc
            if indicator > max_indicator:
                torch.save(
                    model.state_dict(), 
                    os.path.join(cfg.finetuning.save_path, 'best_validation_weights.pt')
                )
                max_indicator = indicator
                print('Best in validation set. Model save at {}'.format(cfg.finetuning.save_path))

        if epoch % cfg.finetuning.save_interval == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(cfg.finetuning.save_path, 'epoch_{}.pt'.format(epoch))
            )

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler:
            lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training accuracy', avg_acc, epoch)
            logger.add_scalar('training kappa', avg_kappa, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    torch.save(
        model.state_dict(), 
        os.path.join(cfg.finetuning.save_path, 'final_weights.pt')
    )

    if logger:
        logger.close()


def evaluate(cfg, model, checkpoint, test_dataset, estimator):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.finetuning.batch_size,
        num_workers=cfg.finetuning.num_workers,
        shuffle=False,
        pin_memory=True
    )

    print('Running on Test set...')
    eval(model, test_loader, estimator, cfg.finetuning.device)

    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
    print('========================================')


def eval(model, dataloader, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X, y = X.to(device), y.to(device).float()

        y_pred = model(X)
        estimator.update(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


def generate_model(cfg):
    clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16')
    visual_encoder = clip.visual

    if cfg.finetuning.checkpoint and cfg.finetuning.checkpoint != 'null':
        weights = torch.load(cfg.finetuning.checkpoint)['state_dict']
        for k in list(weights.keys()):
            prefix = 'module.{}'.format(cfg.finetuning.checkpoint_key)
            if k.startswith(prefix):
                weights[k[len(prefix)+1:]] = weights[k]
            else:
                print(k)
            del weights[k]
        visual_encoder.load_state_dict(weights, strict=True)

    out_features = 1 if cfg.finetuning.criterion == 'mse' else cfg.finetuning.num_classes

    model = Classifier(visual_encoder, 512, out_features, cfg.finetuning.linear)
    model = model.to(cfg.finetuning.device)
    return model


def generate_model_from_solo_learn(cfg):
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    ckpt_path = cfg.finetuning.checkpoint
    if ckpt_path == 'clip':
        # load clip model weights
        import open_clip
        from timm.models.vision_transformer import _convert_openai_clip
        if cfg.finetuning.clip_arch == 'BiomedCLIP':
            clip, _, _ = open_clip.create_model_and_transforms(
                'BiomedCLIP',
                pretrained=cfg.finetuning.clip_pretrained_source,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711]
            )
            visual_encoder = clip.visual.trunk
            state_dict = visual_encoder.state_dict()
        elif cfg.finetuning.clip_arch == 'CLIP':
            clip, _, _ = open_clip.create_model_and_transforms(
                'ViT-B-16',
                pretrained=cfg.finetuning.clip_pretrained_source,
            )
            state_dict = clip.state_dict()
            state_dict = _convert_openai_clip(state_dict, backbone)
    else:
        assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            if "encoder" in k:
                state_dict[k.replace("encoder", "backbone")] = state_dict[k]
            if "backbone" in k:
                state_dict[k.replace("backbone.", "")] = state_dict[k]
            del state_dict[k]

    msg = backbone.load_state_dict(state_dict, strict=False)
    print(msg)
    
    out_features = 1 if cfg.finetuning.criterion == 'mse' else cfg.finetuning.num_classes
    model = Classifier(backbone, 768, out_features, cfg.finetuning.linear)
    model = model.to(cfg.finetuning.device)
    return model

    
def generate_dataset(cfg):
    data_path = cfg.finetuning.data_path

    train_transform, test_transform = data_transforms(cfg)
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_dataset = datasets.ImageFolder(train_path, train_transform, loader=pil_loader)
    test_dataset = datasets.ImageFolder(test_path, test_transform, loader=pil_loader)
    val_dataset = datasets.ImageFolder(val_path, test_transform, loader=pil_loader)

    dataset = train_dataset, test_dataset, val_dataset
    return dataset


def data_transforms(cfg):
    dataset_stats = {
        'ddr': (
            [0.423737496137619, 0.2609460651874542, 0.128403902053833], 
            [0.29482534527778625, 0.20167365670204163, 0.13668020069599152]
        ),
        'aptos2019': (
            [0.46100369095802307, 0.246780663728714, 0.07989078760147095],
            [0.24873991310596466, 0.13842609524726868, 0.08025242388248444]
        ),
        'messidor2': (
            [0.5313033220483331, 0.2531447825309522, 0.08235767606521273],
            [0.2885337192843973, 0.1429464118240488, 0.05240652078759786]
        ),
        'amd': (
            [0.3257227688062399, 0.18428487673823335, 0.08546135457566863],
            [0.1984656808553616, 0.11183091855209067, 0.05459930536715717]
        ),
        'pm': (
            [0.26681682927956496, 0.1443735561146777, 0.06888710368881798],
            [0.18978881498812528, 0.10508514283650022, 0.05144634239746886]
        ),
        'cataract': (
            [0.491061778776388, 0.29281104948897646, 0.1736472586267725],
            [0.2782643222421928, 0.1684671334083035, 0.10083901864192225]
        ),
        'odir5k': (
            [0.4370254420046601, 0.282028658792466, 0.15637293827106252],
            [0.25666372804403526, 0.17451059263511176, 0.10289797228336917]
        ),
        'ss': (
            [0.5552892665515755, 0.32731390326973303, 0.1545033218261648],
            [0.31188307874057447, 0.1881936137287774, 0.09460117233473535]
        ),
        'mpos_cfp': (
            [0.3019424699017467, 0.11050944136388575, 0.017391363520229002],
            [0.16742119442710432, 0.07137858374756204, 0.02951529664023524]
        ),
        'mpos_ffa': (
            [0.3118114544086917, 0.3118114544086917, 0.3118114544086917],
            [0.17192394646211195, 0.17192394646211195, 0.17192394646211195]
        ),
        'octa500': (
            [0.34461964714637966, 0.34461964714637966, 0.34461964714637966],
            [0.1566179334346947, 0.1566179334346947, 0.1566179334346947]
        ),
    }
    mean, std = dataset_stats[cfg.finetuning.dataset]
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(
            size=(cfg.finetuning.input_size, cfg.finetuning.input_size),
            scale=(0.87, 1.15),
            ratio=(0.7, 1.3)
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0
        ),
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ]

    normalization = [
        transforms.Resize((cfg.finetuning.input_size, cfg.finetuning.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    train_preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    test_preprocess = transforms.Compose(normalization)
    return train_preprocess, test_preprocess


class Classifier(nn.Module):
    def __init__(self, encoder, in_features, out_features, linear_eval):
        super(Classifier, self).__init__()
        self.linear_eval = linear_eval
        self.encoder = encoder
        self.head = nn.Linear(in_features, out_features)

        if linear_eval:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.linear_eval:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)

        x = self.head(x)
        return x


class CLIP_Classifier(nn.Module):
    def __init__(self, visual_encoder, text_encoder, tokenizer, dim, text_labels, context_length, device):
        super(CLIP_Classifier, self).__init__()
        self.visual_encoder = visual_encoder
        self.labels = self.parse_labels(text_encoder, tokenizer, text_labels, context_length, device)
        self.head = nn.Linear(dim, dim)

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
    
    def parse_labels(self, text_encoder, tokenizer, text_labels, context_length, device):
        with torch.no_grad():
            tokenized = tokenizer(text_labels, context_length=context_length).to(device)
            text_features = text_encoder(tokenized).to(device)
            text_features = F.normalize(text_features, dim=-1).detach()
        return text_features.to(device)

    def forward(self, x):
        with torch.no_grad():
            x = self.visual_encoder(x)
        x = self.head(x)
        x = x @ self.labels.T
        return x


class Estimator():
    def __init__(self, criterion, num_classes, thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds

        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.cpu()
        predictions = predictions.cpu()
        predictions = self.to_prediction(predictions)

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        return acc

    def get_kappa(self, digits=-1):
        kappa = self.quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def reset(self):
        self.correct = 0
        self.num_samples = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def to_prediction(self, predictions):
        if self.criterion == 'ce':
            predictions = torch.tensor(
                [torch.argmax(p) for p in predictions]
            ).long()
        elif self.criterion == 'mse':
            predictions = torch.tensor(
                [self.classify(p.item()) for p in predictions]
            ).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i

    def quadratic_weighted_kappa(self, conf_mat):
        assert conf_mat.shape[0] == conf_mat.shape[1]
        cate_num = conf_mat.shape[0]

        # Quadratic weighted matrix
        weighted_matrix = np.zeros((cate_num, cate_num))
        for i in range(cate_num):
            for j in range(cate_num):
                weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

        # Expected matrix
        ground_truth_count = np.sum(conf_mat, axis=1)
        pred_count = np.sum(conf_mat, axis=0)
        expected_matrix = np.outer(ground_truth_count, pred_count)

        # Normalization
        conf_mat = conf_mat / conf_mat.sum()
        expected_matrix = expected_matrix / expected_matrix.sum()

        observed = (conf_mat * weighted_matrix).sum()
        expected = (expected_matrix * weighted_matrix).sum()
        return (observed - expected) / (1 - expected)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()