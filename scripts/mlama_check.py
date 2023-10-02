import os
import json
import tqdm
import logging

LOG = logging.getLogger(__name__)


def get_split_range(total_len, split, split_ratio=[8, 1, 1], all_splits=["train", "validation", "test"]):
    split_accumulator = [0]
    ratio_sum_all = sum(split_ratio)
    ratio_sum_cur = 0
    for r in split_ratio:
        ratio_sum_cur += r
        split_accumulator.append(round(total_len * ratio_sum_cur / ratio_sum_all))
    ret = None
    for i, s in enumerate(all_splits):
        if s == split:
            ret = split_accumulator[i], split_accumulator[i + 1]
            break
    assert ret is not None, f"split '{split}' not found"

    if ret[0] >= ret[1]:
        LOG.warning(f"empty split '{split}' with ratio {split_ratio} under total_len {total_len}")
    return ret


data_path_pre = "data/mlama"

languages = "ca az en ar uk fa tr it el ru hr hi sv sq fr ga eu de nl et he es bn ms sr hy ur hu la sl cs af gl fi ro ko cy th be id pt vi ka ja da bg zh pl lv sk lt ta ceb".split()
assert len(languages) == 53

template_name = 'templates.jsonl'

subs = {sp: [] for sp in ['all', "train", "validation", "test"]}

for lg in tqdm.tqdm(languages):
    dataset_dir = os.path.join(data_path_pre, 'mlama1.1', lg)
    for f in tqdm.tqdm(os.listdir(dataset_dir)):
        if f == template_name:
            continue
        dataset_filepath = os.path.join(dataset_dir, f)

        d = []
        with open(dataset_filepath, 'r', encoding='utf8') as fp:
            for l in fp:
                l = l.strip()
                if len(l) > 0:
                    d.append(json.loads(l))

        subs['all'].extend([a['sub_uri'] for a in d])
        for sp in ["train", "validation", "test"]:
            rg = get_split_range(len(d), sp)
            subs[sp].extend([a['sub_uri'] for a in d[rg[0]:rg[1]]])

for k in subs:
    subs[k] = sorted(list(set(subs[k])))
    print(f'{k}: {len(subs[k])}')

# all: 32961
# train: 27225
# validation: 5266
# test: 4603

ex = {sp: [] for sp in ["train", "validation", "test"]}
for k in ["train", "validation", "test"]:
    s = []
    for t in ["train", "test", "validation"]:
        if t == k:
            break
        s.extend(subs[t])
    s = set(s)

    ex[k] = [a for a in subs[k] if a not in s]
    print(f'{k}: {len(ex[k])}')

    with open(f'data/mlama/sub_uri_list/{k}_sub_uri_list.json', 'w', encoding='utf8') as fp:
        json.dump(ex[k], fp, ensure_ascii=True, indent=None)

# train: 27225
# validation: 1820
# test: 3916
