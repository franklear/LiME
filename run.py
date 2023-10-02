import os
import copy
import random
import importlib
import logging

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils


from trainer import EditTrainer
import models


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


@hydra.main(config_path='config', config_name='config')
def run(config):
    if config.debug:
        logging.root.setLevel(logging.DEBUG)
        torch.autograd.set_detect_anomaly(True)

    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)

    if config.task == "lama":
        if config.lime.enabled and config.lime.language_idx is None:
            languages = [
                'ca', 'az', 'en', 'ar', 'uk', 'fa', 'tr', 'it', 'el', 'ru',
                'hr', 'hi', 'sv', 'sq', 'fr', 'ga', 'eu', 'de', 'nl', 'et',
                'he', 'es', 'bn', 'ms', 'sr', 'hy', 'ur', 'hu', 'la', 'sl',
                'cs', 'af', 'gl', 'fi', 'ro', 'ko', 'cy', 'th', 'be', 'id',
                'pt', 'vi', 'ka', 'ja', 'da', 'bg', 'zh', 'pl', 'lv', 'sk',
                'lt', 'ta', 'ceb',
            ]
            config.lime.language_idx = dict([(l, i) for i, l in enumerate(languages)])
        from data_classes.mlama import MLAMADataset

        if config.data.eval_split is None:
            config.data.eval_split = "validation"
        if not config.eval_only or config.alg == "ft" and config.ft.locality.enabled:
            train_set = MLAMADataset(tokenizer, f"{base_dir}/data/mlama", config, subject_entities=False, split="train")
        else:
            train_set = None
        val_set = MLAMADataset(tokenizer, f"{base_dir}/data/mlama", config, subject_entities=False, split=config.data.eval_split)
    elif config.task == "xnli":
        if config.lime.enabled and config.lime.language_idx is None:
            languages = [
                "en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw",
                "th", "tr", "ur", "vi", "zh"
            ]
            config.lime.language_idx = dict([(l, i) for i, l in enumerate(languages)])
        from data_classes.xnli import XNLIDataset

        if not config.eval_only or config.alg == "ft" and config.ft.locality.enabled:
            train_set = XNLIDataset(tokenizer, config, split="train")
        else:
            train_set = None
        if config.data.eval_split is None:
            config.data.eval_split = "validation"
        val_set = XNLIDataset(tokenizer, config, split=config.data.eval_split)
    else:
        raise ValueError(f"Unrecognized task {config.task}")

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    trainer = EditTrainer(alg, config, train_set, val_set)
    if trainer.config.profile:
        with trainer.prof:
            trainer.run()

        trainer.prof.export_chrome_trace(os.path.join(trainer.debug_dir, f"chrome_trace.json"))
        if trainer.prof.with_stack:
            stack_keys = ["self_cpu_time_total"]
            if trainer.config.device in ["cuda"]:
                stack_keys.append("self_cuda_time_total")
            for k in stack_keys:
                trainer.prof.export_stacks(os.path.join(trainer.debug_dir, f"stack_{k}.txt"))
    else:
        trainer.run()


if __name__ == "__main__":
    run()
