import copy
import logging
import os
import random
from itertools import chain, repeat

import datasets
import torch
from torch.utils.data import Dataset

from utils import EditBatchSampler, dict_to, scr

LOG = logging.getLogger(__name__)


class XNLIDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        config,
        split,
    ):
        super().__init__()

        languages = config.languages
        if languages is None:
            # all 15 languages
            languages = [
                "en", "ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw",
                "th", "tr", "ur", "vi", "zh"
            ]
        self.languages = languages
        edit_languages = config.edit_languages
        if edit_languages is None:
            edit_languages = languages.copy()
        assert len(set(edit_languages) - set(languages)) == 0, \
            f"invalid edit_languages {set(edit_languages) - set(languages)}"
        self.edit_languages = edit_languages
        edit_outer_languages = config.edit_outer_languages
        if edit_outer_languages is None:
            edit_outer_languages = languages.copy()
        assert len(set(edit_outer_languages) - set(languages)) == 0, \
            f"invalid edit_outer_languages {set(edit_outer_languages) - set(languages)}"
        if len(set(edit_outer_languages) - set(edit_languages)) > 0:
            LOG.warning(
                f"some edit outer languages are not in edit inner {set(edit_outer_languages) - set(edit_languages)}")
        self.edit_outer_languages = edit_outer_languages
        if len(set(languages) - set(edit_languages) - set(edit_outer_languages)) > 0:
            LOG.warning(f"unused languages {set(languages) - set(edit_languages) - set(edit_outer_languages)}")
        self.language_to_offset = {l: i for i, l in enumerate(set(self.languages))}
        self.offset_to_language = {v: k for k, v in self.language_to_offset.items()}
        self.max_sentence_length = config.max_sentence_length
        self.split = split
        self.tokenizer = tokenizer
        self.config = config

        self.dataset = datasets.load_dataset("xnli", "all_languages", split=self.split, cache_dir=scr())

        if self.config.debug:
            self.dataset = self.dataset.select(indices=range(100), keep_in_memory=True)

        self.base_model_results_path = config.base_model_results_path if self.split == "test" else None
        if self.base_model_results_path is not None:
            self.label_names = self.dataset.features['label'].names
            self.label_name_to_idx = {l: i for i, l in enumerate(self.label_names)}
            self.gold_labels = self.dataset['label']

        raw_len = len(self.dataset)
        self.dataset = self.dataset.map(
            self.preprocess_function_finetuning,
            batched=True,
            keep_in_memory=True,
            load_from_cache_file=False,
            desc=f"Running tokenizer on XNLI {split} dataset",
        )
        assert len(self.dataset) == raw_len * len(self.languages)

        if self.base_model_results_path is not None:
            self.edit_indices_list = {}
            for lg in self.edit_languages:
                lg_result_path = os.path.join(self.base_model_results_path, lg, 'predictions.txt')
                assert os.path.isfile(lg_result_path), f"result file `{lg_result_path}` not exists for language `{lg}`"

                self.edit_indices_list[lg] = []
                with open(lg_result_path, "r", encoding="utf8") as fp:
                    line = fp.readline().strip().split()
                    assert line == ["index", "gold", "prediction"]

                    for i, line in enumerate(fp):
                        line = line.strip().split()
                        assert len(line) == 3
                        assert int(line[0]) == i
                        assert all(s in self.label_names for s in line[2:])
                        if line[1] != self.label_names[self.gold_labels[i]]:
                            LOG.warning(f"gold label mismatch -- sample {i} in language {lg} -- `{line[1]}` in pred file, `{self.label_names[self.gold_labels[i]]}` in dataset")

                        if self.label_names[self.gold_labels[i]] != line[2]:
                            self.edit_indices_list[lg].append(i)

                        if self.config.debug and i + 1 == 100:
                            break
                    assert i + 1 == raw_len

            self.all_edit_indices_list = []
            for lg_offset, lg in enumerate(self.edit_languages):
                lg_edit_indices_list = [i * len(self.edit_languages) + lg_offset for i in self.edit_indices_list[lg]]
                self.all_edit_indices_list.extend(lg_edit_indices_list)
            assert len(self.all_edit_indices_list) == sum(len(l) for l in self.edit_indices_list.values())

    def preprocess_function_finetuning(self, examples):
        # Tokenize the texts
        if isinstance(examples["premise"][0], str):
            return self.tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding=False,
                max_length=self.max_sentence_length,
                truncation=True,
            )

        # for `all_languages`
        languages = self.languages
        premise = list(chain.from_iterable((ex[lg] for lg in languages) for ex in examples["premise"]))
        hyp_idxs = [examples["hypothesis"][0]["language"].index(lg) for lg in languages]
        hypothesis = list(chain.from_iterable((ex["translation"][i] for i in hyp_idxs) for ex in examples["hypothesis"]))

        ret = self.tokenizer(
            premise,
            hypothesis,
            padding=False,
            max_length=self.max_sentence_length,
            truncation=True,
        )

        repeat_times = len(languages)
        ret.update({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": list(chain.from_iterable(repeat(l, repeat_times) for l in examples["label"])),
        })

        return ret

    def __len__(self):
        if self.base_model_results_path is None:
            return len(self.dataset) // len(self.languages) * len(self.edit_languages)
        else:
            return len(self.all_edit_indices_list)

    def __getitem__(self, item):
        q, r = divmod(item, len(self.edit_languages))
        real_idx = q * len(self.languages) + self.language_to_offset[self.edit_languages[r]]
        output = self.dataset[real_idx]

        return output

    def collate_fn(self, batch):
        for d in batch:
            for k in ["hypothesis", "premise"]:
                if k in d:
                    del d[k]

        batch = self.tokenizer.pad(
            batch,
            padding=True if not self.config.debug else "max_length",
            max_length=self.max_sentence_length,
            return_tensors="pt",
        )

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        n = min(n, len(self))
        sampler = EditBatchSampler(
            n,
            n_edits=self.config.data.n_edits,
            memorize_mode=self.config.single_batch,
            seed=self.config.seed,
            indices=self.all_edit_indices_list[:n] if self.base_model_results_path is not None else None,
            loc_n=len(self.dataset) // len(self.languages) * len(self.edit_languages) if self.base_model_results_path is not None else None,
        )
        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            edit_list = [copy.deepcopy(self[idx]) for idx in edit_idxs]

            language_idx_inner = []
            language_idx_outer = []
            if not self.config.lime.enabled or self.config.lime.language_idx is None:
                language_idx_inner = None
                language_idx_outer = None

            outer_list = []
            for edit_idx in edit_idxs:
                data_idx = edit_idx // len(self.edit_languages)
                outer_language = random.choice(self.edit_outer_languages)
                outer_real_idx = data_idx * len(self.languages) + self.language_to_offset[outer_language]

                outer_list.append(copy.deepcopy(self.dataset[outer_real_idx]))

                if language_idx_inner is not None:
                    inner_language = self.offset_to_language[edit_idx % len(self.edit_languages)]
                    language_idx_inner.append(self.config.lime.language_idx[inner_language])
                if language_idx_outer is not None:
                    language_idx_outer.append(self.config.lime.language_idx[outer_language])

            loc_list = []
            for loc_idx in loc_idxs:
                data_idx = loc_idx // len(self.edit_languages)
                loc_language = random.choice(self.edit_outer_languages)
                loc_real_idx = data_idx * len(self.languages) + self.language_to_offset[loc_language]

                loc_list.append(copy.deepcopy(self.dataset[loc_real_idx]))

            for edit, outer, edit_idx in zip(edit_list, outer_list, edit_idxs):
                if self.base_model_results_path is None:
                    new_label = random.randrange(self.dataset.features["label"].num_classes)
                else:
                    new_label = self.gold_labels[edit_idx // len(self.edit_languages)]

                edit["label"] = new_label
                outer["label"] = new_label

            inner_batch = self.collate_fn(edit_list)
            outer_batch = self.collate_fn(outer_list)
            loc_batch = self.collate_fn(loc_list)

            if language_idx_inner is not None:
                language_idx_inner = torch.tensor(language_idx_inner)
            if language_idx_outer is not None:
                language_idx_outer = torch.tensor(language_idx_outer)

            batch = {
                "edit_inner": inner_batch,
                "edit_outer": outer_batch,
                "loc": loc_batch,
                "cond": None,
                "language_idx_inner": language_idx_inner,
                "language_idx_outer": language_idx_outer,
            }

            yield dict_to(batch, self.config.device)
