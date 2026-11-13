#!/usr/bin/env python3
"""The split variants of the experiments (README.md), from the RSL-News release:

    splits.py RSL_NEWS OUT      # RSL_NEWS: manifests/, dataset/ (labels), tokenizer/

Writes OUT/data-<variant>/ (manifests holding the broadcasts the variant uses, `split` set to their role in it; links
to RSL_NEWS/dataset and RSL_NEWS/tokenizer), OUT/splits/subsets.json (each variant's test broadcasts: channel,
interpreter, ...), OUT/splits/<variant>.devsub.json (5,000 dev sentences for decoding choices) and
OUT/splits/hours.json (train hours). A broadcast's hours are its sentences' end - start. Random draws use
random.Random(0) on lists sorted by key, so the result never depends on manifest order."""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHANNELS = {'digi24': 'Digi24', 'prima': 'PrimaTV', 'protv': 'ProTV'}
CHECKSUMS = {'digi24': '2f08af3e2c0ca67564739e8d74073bed8d0c42604f80a897dbc0d5e5cf9918d6',     # the release's manifests
             'prima': '83d6525449355c89436f7cb06b9839259ff9a129649214f274e2cae91652221c',
             'protv': '201d9ec44c28f6b5fefb4cbc44da50463e9bacfc27bb680b612248767a2513aa'}


def load(root):
    """[broadcast]: {manifest, channel, key, item, split, interpreter, date, hours, names}"""
    dates = json.loads((HERE / 'primatv_upload_dates.json').read_text())     # PrimaTV items have no date
    out = []
    for m, channel in CHANNELS.items():
        raw = (root / 'manifests' / f'{m}_manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != CHECKSUMS[m]:
            sys.exit(f'manifests/{m}_manifest.json is not the version these experiments used')
        for item in json.loads(raw)['items']:
            key = item['id'] if m == 'prima' else item['date']
            episode = item['id'] if m == 'prima' else item['date'].replace('-', '/')
            hours, names = 0.0, []
            for s in item['segments']:
                seg = f'{channel}/{episode}/segment_{s["index"]}'
                rows = json.loads((root / 'dataset' / f'{seg}.json').read_text(encoding='utf-8'))
                hours += sum(r['end'] - r['start'] for r in rows) / 3600
                names += [f'{seg}:{i}' for i in range(len(rows))]
            out.append(dict(manifest=m, channel=channel, key=key, item=item, split=item['split'],
                            interpreter=item['interpreter'], date=item.get('date') or dates[item['id']],
                            hours=hours, names=names))
    return out


def ordered(bs):
    return sorted(bs, key=lambda b: (b['channel'], b['key']))


def prefix(bs, share):
    """Per channel: the shortest prefix of the shuffled broadcasts reaching `share` of their hours (nested)."""
    out = []
    for channel in CHANNELS.values():
        mine = ordered(b for b in bs if b['channel'] == channel)
        random.Random(0).shuffle(mine)
        total, acc = sum(b['hours'] for b in mine), 0.0
        for b in mine:
            if acc >= share * total:
                break
            out.append(b); acc += b['hours']
    return out


def periods(bs):
    """Per channel, all broadcasts by (date, key): 'train' while the hours before them are under 90% of the
    channel's, 'buffer' under 95% (dropped), 'future' after."""
    period = {}
    for channel in CHANNELS.values():
        mine = sorted((b for b in bs if b['channel'] == channel), key=lambda b: (b['date'], b['key']))
        total, acc = sum(b['hours'] for b in mine), 0.0
        for b in mine:
            period[id(b)] = 'train' if acc < 0.90 * total else 'buffer' if acc < 0.95 * total else 'future'
            acc += b['hours']
    return period


def variants(bs, period):
    """{variant: {id(broadcast): 'train' | 'val' | 'test'}}"""
    train, val, test = ([b for b in bs if b['split'] == s] for s in ('train', 'val', 'test'))
    roles = lambda tr, va, te: {**{id(b): 'train' for b in tr}, **{id(b): 'val' for b in va}, **{id(b): 'test' for b in te}}
    v = {'std': roles(train, val, test)}
    for name, share in (('lc12', 0.125), ('lc25', 0.25), ('lc50', 0.5)):
        v[name] = roles(prefix(train, share), val, test)
    i67 = lambda b: b['interpreter'] in ('I6', 'I7')
    v['hold-i6i7'] = roles([b for b in train if not i67(b)], [b for b in val if not i67(b)], test + [b for b in bs if i67(b)])
    v['hold-i2'] = roles([b for b in train if b['interpreter'] != 'I2'], [b for b in val if b['interpreter'] != 'I2'], test)
    target, pool = sum(b['hours'] for b in train if b['interpreter'] == 'I2'), ordered(b for b in train if b['interpreter'] != 'I2')
    random.Random(0).shuffle(pool)
    drop, acc = set(), 0.0          # other signers' broadcasts totalling I2's train hours, +- 1 h
    for b in pool:
        if acc >= target - 1:
            break
        if acc + b['hours'] <= target + 1:
            drop.add(id(b)); acc += b['hours']
    v['hold-i2-ctrl'] = roles([b for b in train if id(b) not in drop], val, test)
    v['chrono'] = roles([b for b in train if period[id(b)] == 'train'], [b for b in val if period[id(b)] == 'train'],
                        [b for b in test if period[id(b)] == 'train'] + [b for b in bs if period[id(b)] == 'future'])
    for channel in CHANNELS.values():
        v[f'hold-{channel.lower()}'] = roles([b for b in train if b['channel'] != channel],
                                             [b for b in val if b['channel'] != channel], test)
    return v


def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    bs = load(root)
    period = periods(bs)
    (out / 'splits').mkdir(parents=True, exist_ok=True)
    subsets, hours, devsubs = {}, {}, {}
    for name, roles in variants(bs, period).items():
        d = out / f'data-{name}'
        (d / 'manifests').mkdir(parents=True, exist_ok=True)
        for m in CHANNELS:
            src = json.loads((root / 'manifests' / f'{m}_manifest.json').read_text())
            items = [{**b['item'], 'split': roles[id(b)]} for b in bs if b['manifest'] == m and id(b) in roles]
            (d / 'manifests' / f'{m}_manifest.json').write_text(json.dumps({**src, 'items': items}, ensure_ascii=False, indent=1))
        for link in ('dataset', 'tokenizer'):
            (d / link).unlink(missing_ok=True)
            (d / link).symlink_to(root / link)
        hours[name] = sum(b['hours'] for b in bs if roles.get(id(b)) == 'train')
        subsets[name] = {f'{b["channel"]}/{b["key"]}': {
            'channel': b['channel'], 'interpreter': b['interpreter'], 'in_v1_test': b['split'] == 'test',
            'i6i7': b['interpreter'] in ('I6', 'I7'),
            'period': {'train': 'within', 'buffer': 'buffer', 'future': 'future'}[period[id(b)]]}
            for b in bs if roles.get(id(b)) == 'test'}
        dev = tuple(sorted(id(b) for b in bs if roles.get(id(b)) == 'val'))
        if dev not in devsubs:
            names = sorted(n for b in bs if roles.get(id(b)) == 'val' for n in b['names'])
            devsubs[dev] = sorted(random.Random(0).sample(names, min(5000, len(names))))
        (out / 'splits' / f'{name}.devsub.json').write_text(json.dumps(devsubs[dev]))
        print(f'{name:13s} train {hours[name]:7.1f} h')
    (out / 'splits' / 'subsets.json').write_text(json.dumps(subsets, indent=1))
    (out / 'splits' / 'hours.json').write_text(json.dumps(hours, indent=1))


if __name__ == '__main__':
    main()
