# CiLA: CLIP-informed Lesion Attribute Learning

This is the pytorch implementation of the paper submitted to MICCAI 2025:

> Anonymous, "Enhancing Fundus Self-supervised Learning via CLIP-informed Lesion Attribute Learning", under review, 2025.

## Dataset

The datasets used in this work are listed as follows:

Pretraining:

- EyePACS [[homepage](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview)].

Evalutation for classification:

- Messidor-2 [[images](https://www.adcis.net/en/third-party/messidor2/)] [[labels](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)].
- IChallenge-AMD [[homepage](https://refuge.grand-challenge.org/iChallenge-AMD/)]
- Retinal [[homepage](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)]
- ODIR-5k [[homepage](https://odir2019.grand-challenge.org)]
- MPOS [[homepage](https://github.com/whq-xxh/FFA-Synthesis)]

## Installation

To install the dependencies, download the repository and run:

```shell
conda create -n cila python=3.8.0
conda activate cila
pip install .
```

## Usage

### Pre-training

1\. Organize the EyePACS in the following format:

```
├── dataset
    ├── class1
        ├── image1.jpg
        ├── image2.jpg
        ├── ...
    ├── class2
        ├── image3.jpg
        ├── image4.jpg
        ├── ...
    ├── class3
    ├── ...
```

> **Note:** Image labels are not used in the pretraining stage.

2\. Modify the `train_path` in `/scripts/pretrain/eyepacs/[SSL method].yaml` to point to your dataset. You can also adjust the training device by modifying `devices` in this YAML file.

3\. Run pre-training:

```shell
python main_pretrain.py --config-path scripts/pretrain/eyepacs/ --config-name [SSL method].yaml
```

### Evaluation

1\. Ensure each dataset follows this structure:

```
├── dataset
    ├── train
        ├── class1
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
        ├── class2
            ├── image3.jpg
            ├── image4.jpg
            ├── ...
        ├── class3
        ├── ...
    ├── val
    ├── test
```

2\. Run fine-tuning on the downstream dataset with:

```shell
python finetuning.py --config-path ./scripts/finetune/fundus/ --config-name finetuning.yaml \
    ++finetuning.dataset={} ++finetuning.data_path={} ++finetuning.num_classes={} \
    ++finetuning.save_path={} ++finetuning.checkpoint={} ++finetuning.learning_rate={}
```

You can either update the arguments in `{}` directly or modify the configuration file `./scripts/finetune/fundus/finetuning.yaml`.

3\. Linear evaluation on downstream dataset by:

```shell
python finetuning.py --config-path ./scripts/finetune/fundus/ --config-name finetuning.yaml \
    ++finetuning.dataset={} ++finetuning.data_path={} ++finetuning.num_classes={} \
    ++finetuning.save_path={} ++finetuning.checkpoint={} ++finetuning.learning_rate={} \
    ++finetuning.linear=true
```

## Acknowledgment

This repository is based on [solo-learn](https://github.com/vturrisi/solo-learn), a wonderful self-supervised learning framework for PyTorch.
