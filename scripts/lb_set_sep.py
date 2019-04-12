import json

with open('../data/labelboxout/all_skip.json') as ms:
    sm = json.load(ms)

for ds in set([ds['Dataset Name'] for ds in sm]):
    dss = [d for d in sm if d['Dataset Name']==ds]
    with open(f'{ds}_skip.json','w') as jd:
        json.dump(dss,jd)
