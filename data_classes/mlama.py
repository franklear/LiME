import json
import logging
import os
import random
import copy

import tqdm
import torch
from torch.utils.data import Dataset

from utils import EditBatchSampler, dict_to

LOG = logging.getLogger(__name__)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def filter_text(iterator):
    valid = []
    for text in iterator:
        if len(text.split(' ')) < 50:
            continue
        if not is_ascii(text):
            continue
        valid.append(text)

    return valid


class MLAMADataset(Dataset):
    def __init__(
        self,
        tokenizer: "transformers.tokenization_utils.PretrainedTokenizer",
        data_path_pre: str,
        config,
        subject_entities: bool,
        split,
    ):
        super().__init__()

        languages = config.languages
        if languages is None:
            # all 53 languages
            languages = [
                'ca', 'az', 'en', 'ar', 'uk', 'fa', 'tr', 'it', 'el', 'ru',
                'hr', 'hi', 'sv', 'sq', 'fr', 'ga', 'eu', 'de', 'nl', 'et',
                'he', 'es', 'bn', 'ms', 'sr', 'hy', 'ur', 'hu', 'la', 'sl',
                'cs', 'af', 'gl', 'fi', 'ro', 'ko', 'cy', 'th', 'be', 'id',
                'pt', 'vi', 'ka', 'ja', 'da', 'bg', 'zh', 'pl', 'lv', 'sk',
                'lt', 'ta', 'ceb',
            ]
        self.languages = languages
        edit_languages = config.edit_languages
        if edit_languages is None:
            edit_languages = languages.copy()
        assert len(set(edit_languages) - set(languages)) == 0,\
            f"invalid edit_languages {set(edit_languages) - set(languages)}"
        self.edit_languages = edit_languages
        edit_outer_languages = config.edit_outer_languages
        if edit_outer_languages is None:
            edit_outer_languages = languages.copy()
        assert len(set(edit_outer_languages) - set(languages)) == 0,\
            f"invalid edit_outer_languages {set(edit_outer_languages) - set(languages)}"
        if len(set(edit_outer_languages) - set(edit_languages)) > 0:
            LOG.warning(f"some edit outer languages are not in edit inner {set(edit_outer_languages) - set(edit_languages)}")
        self.edit_outer_languages = edit_outer_languages
        if len(set(languages) - set(edit_languages) - set(edit_outer_languages)) > 0:
            LOG.warning(f"unused languages {set(languages) - set(edit_languages) - set(edit_outer_languages)}")
        self.max_sentence_length = config.max_sentence_length
        self.tokenizer = tokenizer
        self.config = config
        self.current_relation = None
        self.subject_entities = subject_entities
        self.split = split
        self.split_by = config.split_by
        assert config.split_by in [None, "template", "sample"]

        all_splits = ["train", "validation", "test"]
        assert split in all_splits, f"valid splits: {all_splits}, '{split}' not found"
        self.all_splits = all_splits

        assert len(config.split_ratio) == len(all_splits)
        self.split_ratio = config.split_ratio

        self.valid_sub_uris = None
        if config.sub_uri_list_dir is not None:
            valid_sub_uri_list_path = os.path.join(config.sub_uri_list_dir, f'{self.split}_sub_uri_list.json')
            assert os.path.isfile(valid_sub_uri_list_path), f'cannot find sub_uri_list file {valid_sub_uri_list_path}'
            with open(valid_sub_uri_list_path, 'r', encoding='utf8') as fp:
                self.valid_sub_uris = set(json.load(fp))
            LOG.warning(f'loaded sub_uri_list from {valid_sub_uri_list_path}, len = {len(self.valid_sub_uris)}')
            assert self.split == 'train' and len(self.valid_sub_uris) == 27225 \
                or self.split == 'validation' and len(self.valid_sub_uris) == 1820 \
                or self.split == 'test' and len(self.valid_sub_uris) == 3916

        # relations in every f"{data_path_pre}/mlama1.1/{lg}/templates.jsonl"
        # following the origin order in mLAMA dataset
        all_relations = [
            'P19', 'P20', 'P279', 'P37', 'P413', 'P166', 'P449', 'P69', 'P47', 'P138',
            'P364', 'P54', 'P463', 'P101', 'P1923', 'P106', 'P527', 'P102', 'P530', 'P176',
            'P27', 'P407', 'P30', 'P178', 'P1376', 'P131', 'P1412', 'P108', 'P136', 'P17',
            'P39', 'P264', 'P276', 'P937', 'P140', 'P1303', 'P127', 'P103', 'P190', 'P1001',
            'P31', 'P495', 'P159', 'P36', 'P740', 'P361',
            'date_of_birth', 'place_of_birth', 'place_of_death'
        ]
        assert len(all_relations) == 49
        # excluding relations that actually not exist in mLAMA
        relations_to_exclude = ['P166', 'P69', 'P54', 'P1923', 'P102']
        # excluding relations where only an empty file exist in too many (53, 13, 13, resp.) languages
        relations_to_exclude.extend(['date_of_birth', 'place_of_birth', 'place_of_death'])
        # keeping relations where only an empty file exist in few (1, 1, 4, 2, 1, resp.) languages
        # relations_to_exclude.extend(['P108', 'P136', 'P264', 'P413', 'P449'])
        all_relations = [s for s in all_relations if s not in relations_to_exclude]
        self.relations_to_exclude = relations_to_exclude
        self.all_relations = all_relations
        assert len(self.all_relations) == 41
        self.split_relations = all_relations

        if config.split_by == "template":
            split_range = self.get_split_range(total_len=len(all_relations))
            self.split_relations = all_relations[split_range[0]:split_range[1]]
            if self.config.debug and self.split_ratio == [8, 1, 1] and self.config.split_by == "template":
                assert self.split == "train" and len(self.split_relations) == 33 or \
                       self.split == "validation" and len(self.split_relations) == 4 or \
                       self.split == "test" and len(self.split_relations) == 4

        self.data = {}
        self.uuid_to_sample = {}
        self.relation_uris = {}
        for language in tqdm.tqdm(sorted(languages, key=lambda l: l not in edit_outer_languages), desc="languages"):
            templates_filename = f"{data_path_pre}/mlama1.1/{language}/templates.jsonl"
            with open(templates_filename, "r", encoding="utf8") as fp:
                templates = [json.loads(line) for line in fp]

            candidates = {}
            for candidates_source in ("TREx_multilingual_objects", "GoogleRE_objects"):
                candidates_path = f"{data_path_pre}/{candidates_source}/{language}.json"
                with open(candidates_path, "r", encoding="utf8") as fp:
                    candidates.update(json.load(fp))

            self.data[language] = {
                "templates": [],
                "candidates": candidates,
                "uri_to_label": {},
                "label_to_uri": {},
                "dicts_num_mask": {},
                "dicts_num_mask_all": {},
                "relations": {},
                "all_samples": {},
            }

            uri_to_label = {}
            exclude_labels = set()
            exclude_samples = []
            for idx_template, template in tqdm.tqdm(enumerate(templates), desc="templates", total=len(templates)):
                LOG.debug(f"template[{idx_template}] = {json.dumps(template, indent=4, ensure_ascii=False)}")

                relation = template["relation"]
                assert relation in self.all_relations or relation in self.relations_to_exclude, \
                    f"relation '{relation}' not exists in both {self.all_relations} and {self.relations_to_exclude}"
                if relation not in self.split_relations:
                    LOG.info(f"skip relation '{relation}' in split '{self.split}'")
                    continue

                dataset_filename = f"{data_path_pre}/mlama1.1/{language}/{relation}.jsonl"
                try:
                    with open(dataset_filename, "r", encoding="utf8") as fp:
                        relations = [json.loads(line) for line in fp]
                except Exception as e:
                    LOG.warning(f"Relation {relation} excluded. Exception: {e}")
                    continue

                if len(relations) == 0 or len(candidates[relation]["objects"]) == 0:
                    LOG.warning(f"Empty relation {relation} excluded.")
                    continue

                for i, r in enumerate(relations, start=1):
                    if "uuid" not in r:
                        r["uuid"] = str(r["lineid"]) if "lineid" in r else str(-i)

                candidates_max_length = max(len(tokenizer.tokenize(obj)) for obj in candidates[relation]["objects"])
                dict_num_mask = {l: {} for l in range(1, candidates_max_length + 1)}
                for obj in candidates[relation]["objects"]:
                    tokens = tokenizer.tokenize(obj)
                    dict_num_mask[len(tokens)][obj] = tokenizer.convert_tokens_to_ids(tokens)
                dict_num_mask = {k: v for k, v in dict_num_mask.items() if len(v) > 0}

                relations, samples_exluded, filter_msg = self.filter_samples(
                    samples=relations,
                    max_sentence_length=config.max_sentence_length,
                    template=template["template"],
                )
                if samples_exluded > 0:
                    LOG.warning(filter_msg)

                if config.split_by == "sample":
                    split_range = self.get_split_range(total_len=len(relations))
                    relations = relations[split_range[0]:split_range[1]]

                if len(relations) == 0:
                    LOG.warning(f"Empty relation {relation} excluded in split {self.split}.")
                    continue

                if relation not in self.relation_uris:
                    self.relation_uris[relation] = set()

                facts = set()
                for sample in relations:
                    sub_uri = sample["sub_uri"]
                    if self.valid_sub_uris is not None and sub_uri not in self.valid_sub_uris:
                        continue
                    obj_uri = sample["obj_uri"]
                    sub = sample["sub_label"]
                    obj = sample["obj_label"]
                    uuid = sample["uuid"]
                    if subject_entities:
                        uri_labels = ((sub_uri, sub),)
                        self.relation_uris[relation].add(sub_uri)
                        exclude_labels.add(obj)
                    else:
                        uri_labels = ((obj_uri, obj),)
                        self.relation_uris[relation].add(obj_uri)
                    for uri, label in uri_labels:
                        if uri not in uri_to_label:
                            uri_to_label[uri] = label
                        assert label == uri_to_label[uri]
                    if (sub, obj, uuid) not in facts:
                        facts.add((sub, obj, uuid))
                LOG.debug("distinct template facts: {}".format(len(facts)))

                all_samples = []
                ex_samples = []
                for fact in facts:
                    (sub, obj, uuid) = fact
                    sample = {
                        "sub_label": sub, "obj_label": obj, "uuid": uuid,
                        "template": template["template"]
                    }
                    # substitute all sentences with a standard template
                    sample["masked_sentences"] = self.parse_template(
                        template=template["template"].strip(),
                        subject_label=sample["sub_label"].strip(),
                        object_label=self.tokenizer.mask_token,
                    )
                    sample["language"] = language
                    sample["relation"] = relation

                    if language not in edit_outer_languages and len(self.uuid_to_sample.get(uuid, {})) < 1:
                        ex_samples.append(sample)
                        continue
                    else:
                        all_samples.append(sample)

                    if uuid not in self.uuid_to_sample:
                        self.uuid_to_sample[uuid] = {}
                    self.uuid_to_sample[uuid][language] = sample

                LOG.debug(f"distinct template facts: {len(facts)}"
                          f" exclude samples: {len(ex_samples)}"
                          f" samples: {len(all_samples)}")
                assert len(ex_samples) + len(all_samples) == len(facts)
                exclude_samples.extend(ex_samples)

                self.data[language]["templates"].append(template)
                self.data[language]["relations"][relation] = relations
                self.data[language]["all_samples"][relation] = all_samples

                self.data[language]["dicts_num_mask"][relation] = dict_num_mask
                for l in dict_num_mask:
                    if l not in self.data[language]["dicts_num_mask_all"]:
                        self.data[language]["dicts_num_mask_all"][l] = {}
                    self.data[language]["dicts_num_mask_all"][l].update(dict_num_mask[l])

            self.data[language]["dicts_num_mask_all"] = dict(sorted(
                list(self.data[language]["dicts_num_mask_all"].items())
            ))

            raw_uri_to_label = uri_to_label
            uri_to_label = {k: v for k, v in uri_to_label.items() if v not in exclude_labels}
            self.data[language]["uri_to_label"] = dict(
                sorted(
                    list(uri_to_label.items()),
                    key=lambda t: (not t[0].startswith("Q"), int(t[0][1:]) if t[0].startswith("Q") else t[0])
                )
            )
            self.data[language]["label_to_uri"] = {v: k for k, v in self.data[language]["uri_to_label"].items()}
            LOG.debug(
                f"raw {'subject' if subject_entities else 'object'} uris: {len(raw_uri_to_label)}"
                f" exclude {'object' if subject_entities else '(n/a)'} labels: {len(exclude_labels)}"
                f" final {'subject' if subject_entities else 'object'} uris: {len(self.data[language]['uri_to_label'])}"
                f" final {'subject' if subject_entities else 'object'} labels: {len(self.data[language]['label_to_uri'])}"
                f" exclude samples: {len(exclude_samples)}"
                f" all samples: {sum(len(s) for s in self.data[language]['all_samples'].values())}"
            )
            if not subject_entities:
                assert len(exclude_labels) == 0
            if language in edit_outer_languages:
                assert len(exclude_samples) == 0

    def get_split_range(self, total_len):
        split_accumulator = [0]
        ratio_sum_all = sum(self.split_ratio)
        ratio_sum_cur = 0
        for r in self.split_ratio:
            ratio_sum_cur += r
            split_accumulator.append(round(total_len * ratio_sum_cur / ratio_sum_all))
        ret = None
        for i, s in enumerate(self.all_splits):
            if s == self.split:
                ret = split_accumulator[i], split_accumulator[i + 1]
                break
        assert ret is not None, f"split '{self.split}' not found"
        LOG.debug(f"{self.split=} {self.split_ratio=} {split_accumulator=} {ratio_sum_all=} {ratio_sum_cur=} {total_len=}")

        if ret[0] >= ret[1]:
            LOG.warning(f"empty split '{self.split}' with ratio {self.split_ratio} under total_len {total_len}")
        return ret

    def filter_samples(self, samples, max_sentence_length, template, vocab_subset=None):
        """
        from mlama/scripts/batch_eval_KB_completion_mBERT_ranked.py
        with few fixes
        """
        msg = ""
        new_samples = []
        samples_exluded = 0
        for sample in samples:
            if "obj_label" in sample and "sub_label" in sample:

                obj_label_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(sample["obj_label"])
                )

                # if obj_label_ids:
                #     recostructed_word = " ".join(
                #         [model.vocab[x] for x in obj_label_ids]
                #     ).strip()
                # else:
                #     recostructed_word = None

                excluded = False
                if not template or len(template) == 0:
                    masked_sentences = sample["masked_sentences"]
                    text = " ".join(masked_sentences)
                    if len(text.split()) > max_sentence_length:
                        msg += "\tEXCLUDED for exeeding max sentence length: {}\n".format(
                            masked_sentences
                        )
                        samples_exluded += 1
                        excluded = True
                """if sample['from_english']:
                    msg += "\tEXCLUDED not in language \n"
                    excluded = True
                    samples_exluded += 1"""
                # MAKE SURE THAT obj_label IS IN VOCABULARIES
                if vocab_subset:
                    for x in sample["obj_label"].split(" "):
                        if x not in vocab_subset:
                            excluded = True
                            msg += "\tEXCLUDED object label {} not in vocab subset\n".format(
                                sample["obj_label"]
                            )
                            samples_exluded += 1
                            break
                if excluded:
                    pass
                elif obj_label_ids is None:
                    msg += "\tEXCLUDED object label is {} None\n".format(
                        sample["obj_label"]
                    )
                    samples_exluded += 1

                #   samples_exluded+=1
                elif "judgments" in sample:
                    # only for Google-RE
                    num_no = 0
                    num_yes = 0
                    for x in sample["judgments"]:
                        if x["judgment"] == "yes":
                            num_yes += 1
                        else:
                            num_no += 1
                    if num_no > num_yes:
                        # SKIP NEGATIVE EVIDENCE
                        msg += "\tEXCLUDED negative evidence for num_no({}) > num_yes({})\n".format(
                            num_no, num_yes
                        )
                        samples_exluded += 1
                        pass
                    else:
                        new_samples.append(sample)
                else:
                    new_samples.append(sample)
            else:
                msg += "\tEXCLUDED since 'obj_label' not sample or 'sub_label' not in sample: {}\n".format(
                    sample
                )
                samples_exluded += 1
        msg += "samples exluded  : {}\n".format(samples_exluded)
        # LOG.info(msg)
        return new_samples, samples_exluded, msg

    def parse_template(self, template, subject_label, object_label):
        """
        from mlama/scripts/batch_eval_KB_completion_mBERT_ranked.py
        with few fixes
        """
        SUBJ_SYMBOL = "[X]"
        OBJ_SYMBOL = "[Y]"
        template = template.replace(SUBJ_SYMBOL, subject_label)
        template = template.replace(OBJ_SYMBOL, object_label)
        return [template]

    def set_relation(self, relation: str):
        not_exist_list = [l for l in self.languages if relation not in self.data[l]["relations"]]
        if len(not_exist_list) > 0:
            LOG.warning(
                f"relation {relation} not exists in the following languages"
                f" ({len(not_exist_list)} out of {len(self.languages)}):"
                f"\n{not_exist_list}"
            )
        assert len(not_exist_list) < len(self.languages)
        self.current_relation = relation

    def collate_fn(self, input_list):
        sentences = []
        for d in input_list:
            if "label_length" not in d:
                d["label_length"] = len(self.tokenizer.tokenize(d["label"]))

            sentence = d["masked_sentences"][0].replace(
                self.tokenizer.mask_token, " ".join([self.tokenizer.mask_token] * d["label_length"])
            )
            sentences.append(sentence)

        batch = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        if "label" in input_list[0]:
            labels = batch["input_ids"].clone()
            labels[labels != self.tokenizer.mask_token_id] = -100
            for i, d in enumerate(input_list):
                labels[i][labels[i] == self.tokenizer.mask_token_id] = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(d["label"]))
                    if "label" in d else -100
                )
            batch["labels"] = labels

        return batch

    def edit_generator(self, batch_size, n=None, use_origin_loc_obj=None, use_translated_labels=None):
        if n is None:
            n = len(self)
        n = min(n, len(self))
        if use_origin_loc_obj is None:
            use_origin_loc_obj = False
        if use_translated_labels is None:
            use_translated_labels = self.config.use_translated_labels
        sampler = EditBatchSampler(
            n=n,
            n_edits=self.config.data.n_edits,
            memorize_mode=self.config.single_batch,
            loc_disjoint=True,
            seed=self.config.seed,
        )
        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)

            edit_list = [copy.deepcopy(self[idx]) for idx in edit_idxs]
            if self.config.use_translate:
                outer_list = []
                for d in edit_list:
                    translates = self.uuid_to_sample[d["uuid"]]
                    translates = {k: v for k, v in translates.items() if k in self.edit_outer_languages}
                    language, translate = random.choice(list(translates.items()))
                    translate = copy.deepcopy(translate)
                    outer_list.append(translate)
            else:
                outer_list = copy.deepcopy(edit_list)

            if self.config.use_translate:
                loc_list = []
                for d_idx in loc_idxs:
                    d = copy.deepcopy(self[d_idx])
                    translates = self.uuid_to_sample[d["uuid"]]
                    translates = {k: v for k, v in translates.items() if k in self.edit_outer_languages}
                    language, translate = random.choice(list(translates.items()))
                    translate = copy.deepcopy(translate)
                    loc_list.append(translate)
            else:
                loc_list = [copy.deepcopy(self[idx]) for idx in loc_idxs]

            for edit, outer in zip(edit_list, outer_list):
                valid_uri_list = [
                    u for u in self.relation_uris[edit["relation"]]
                    if u in self.data[edit["language"]]["uri_to_label"]
                ]

                if use_translated_labels:
                    is_valid_obj = False
                    while not is_valid_obj:
                        obj_uri = random.choice(valid_uri_list)
                        if obj_uri in self.data[outer["language"]]["uri_to_label"]:
                            is_valid_obj = True

                    edit["label"] = self.data[edit["language"]]["uri_to_label"][obj_uri]
                    outer["label"] = self.data[outer["language"]]["uri_to_label"][obj_uri]
                else:
                    obj_uri = random.choice(valid_uri_list)

                    label = self.data[edit["language"]]["uri_to_label"][obj_uri]
                    edit["label"] = label
                    outer["label"] = label

                    outer["masked_sentences"] = self.parse_template(
                        template=outer["template"].strip(),
                        subject_label=edit["sub_label"].strip(),
                        object_label=self.tokenizer.mask_token,
                    )

            for loc in loc_list:
                if use_origin_loc_obj:
                    loc["label"] = loc["obj_label"]
                else:
                    loc["label_length"] = random.choices(
                        list(self.data[loc["language"]]["dicts_num_mask_all"].keys()),
                        weights=[len(v) for v in self.data[loc["language"]]["dicts_num_mask_all"].values()],
                    )[0]

            if not self.config.lime.enabled or self.config.lime.language_idx is None:
                language_idx_inner = None
                language_idx_outer = None
            else:
                language_idx_inner = [self.config.lime.language_idx[s["language"]] for s in edit_list]
                language_idx_outer = [self.config.lime.language_idx[s["language"]] for s in outer_list]
                language_idx_inner = torch.tensor(language_idx_inner)
                language_idx_outer = torch.tensor(language_idx_outer)

            inner_batch = self.collate_fn(edit_list)
            outer_batch = self.collate_fn(outer_list)
            loc_batch = self.collate_fn(loc_list)

            batch = {
                "edit_inner": inner_batch,
                "edit_outer": outer_batch,
                "loc": loc_batch,
                "cond": None,
                "language_idx_inner": language_idx_inner,
                "language_idx_outer": language_idx_outer,
            }

            yield dict_to(batch, self.config.device)

    def __len__(self):
        return sum(sum(len(r) for r in self.data[l]["all_samples"].values()) for l in self.edit_languages)

    def __getitem__(self, idx):
        for language in self.edit_languages:
            lens = [len(r) for r in self.data[language]["all_samples"].values()]
            total_len = sum(lens)
            if total_len <= idx:
                idx -= total_len
                continue

            for s in self.data[language]["all_samples"].values():
                if len(s) <= idx:
                    idx -= len(s)
                    continue

                return s[idx]
