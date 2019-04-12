import os
import json

with open('neg.txt','w') as neg:
    neg.write('')
with open('pos.lst','w') as pos:
    pos.write('')

for i in os.listdir('/Volumes/ElementsExternal/coca_coda/train/ims/not_coke'):
    if i.endswith('.png'):
        fp = './not_coke/'+i
        with open('neg.txt','a') as neg:
            neg.write(fp+'\n')
    if int(i.split('.')[0][3:])%1000==0: print(i)

with open('~/Documents/Github/coca_coda/data/labelboxout/train_labels.json','r') as jl:
    lj=json.load(jl)
with open('~/Documents/Github/coca_coda/data/labelboxout/train_labels_coco.json','r') as jc:
    cj=json.load(jc)

for i in os.listdir('/Volumes/ElementsExternal/coca_coda/train/ims/coke/'):
    if i.endswith('.png'):
        fp = './coke/'+i
        # print(i)
        im_id = [ii['ID'] for ii in lj if ii['External ID']==i][0]
        # print(im_id)
        try:
            bb = [ii['bbox'] for ii in cj['annotations'] if ii['image_id']==im_id][0]
        except:
            print('error with '+i+', '+im_id)
            # print(bb)
        with open('pos.lst','a') as pos:
            pos.write(f'{fp} 1 {int(bb[0])} {int(bb[1])} {int(bb[2])} {int(bb[3])}\n')
        if int(i.split('.')[0][3:])%1000==0: print(i)
