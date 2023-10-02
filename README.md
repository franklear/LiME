# Official code for [Language Anisotropic Cross-Lingual Model Editing](https://arxiv.org/abs/2205.12677)

## Runtime Environment

The code was developed with Python 3.6 + torch 1.10, newer version like Python 3.10 + torch 1.12 should also work.

After installing Python and torch, you can run the command below to install the other dependencies.

```shell
pip install -r requirements.txt
```

## Data

You can download the data we use in the paper on the [🤗Datasets page](https://huggingface.co/datasets/yxu/LiME_data).
The data are placed in the `data` folder by default.

Since all data processing progress is online, we only need to provide the necessary raw data. The data contains two parts.

1. mLAMA: this part contains the raw [mLAMA dataset](https://github.com/norakassner/mlama) along with our split of entities to avoid leaking (please refer to `scripts/mlama_check` for more details).

2. XNLI: this part is the base model we use to edit which is finetuned on XNLI. The raw dataset will be downloaded from [🤗Datasets](https://huggingface.co/datasets/xnli).

## How to Run

A training command is like

```shell
python -m run \
   +alg=mend \
   +experiment=lama \
   +model=mbert-base \
   lime.enabled=True \
   val_interval=20000 \
   model_save_pt=20000 \
   early_stop_patience=40000 \
   seed=0 \
   languages="[\"en\"]"
```

- `alg` means the base editing algorithm. Valid choices include `ft`, `mend`, `efk`, and `enn`.
- `experiment` and `model` specifies the dataset. `+experiment=lama +model=mbert-base` for mLAMA, `+experiment=xnli +model=mbert-base-seqcls` for XNLI.
- `lime.enabled` is the switch of language anisotropic editing, `False` for baseline.
- `languages` is a list passed through a string, containing the languages to use. For example, if we set `languages="[\"en\"]"`, we will train an English monolingual editor. For `languages="[\"en\", \"fr\"]"`, we will train an cross lingual editor between English and French. If we don't give the `languages` parameter, the default behavior is to use all the languages in the dataset, which is the all -> all setting.

The configs are powered by [Hydra](https://hydra.cc), please refer to the `config` folder for more detailed running configs.

## Acknowledgement

The code is developed based on [the official codebase of MEND](https://github.com/eric-mitchell/mend).
In addition to adding language anisotropic cross lingual model editing, we also optimize the code by replacing BatchNorm with LayerNorm, and freezing unchanged parameters.
We try to keep backward compatibility, thus other models and datasets in MEND can also be added to our code.

If you are interested in our work or find issues about the code, feel free to email me through [yxu@ir.hit.edu.cn](mailto:yxu@ir.hit.edu.cn).

You can cite our [paper](https://arxiv.org/abs/2205.12677) as follows

```bibtex
@inproceedings{DBLP:conf/acl/XuHCZ23,
  author       = {Yang Xu and
                  Yutai Hou and
                  Wanxiang Che and
                  Min Zhang},
  editor       = {Anna Rogers and
                  Jordan L. Boyd{-}Graber and
                  Naoaki Okazaki},
  title        = {Language Anisotropic Cross-Lingual Model Editing},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023,
                  Toronto, Canada, July 9-14, 2023},
  pages        = {5554--5569},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.343},
  doi          = {10.18653/v1/2023.findings-acl.343},
  timestamp    = {Thu, 10 Aug 2023 12:35:56 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/XuHCZ23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
