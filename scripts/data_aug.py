import numpy as np
import pandas as pd
import imageio
import os
import ast
import copy
import matplotlib.pyplot as plt
import cv2
import PIL
from PIL import Image
from scipy.ndimage import zoom
import json

def resize(ims,labels):
    resized=[]
    rszlabs=[]
    for im in range(ims.shape[0]):
        # print(len(im[0,197:]))
        if sum([ch for p in ims[im][0,197:] for ch in p])==0:
            img=ims[im][:,28:196]
            imr=Image.fromarray(img)
            imr=imr.resize((302,403), PIL.Image.ANTIALIAS)
            # imr = zoom(im,(403,302,3))
            new_lab=labels[im]
            resized.append(img)
        else:
            imr=ims[im]
            imr=Image.fromarray(ims[im])
            imr=imr.resize((168,224), PIL.Image.ANTIALIAS)
            # imr=zoom(im,(224,168,3))
            labnp = np.zeros((ims[im].shape[0],ims[im].shape[1]))
            for coord in labels[im]:
                labnp[coord['y'],coord['x']]=1
            labnp = cv2.dilate(labnp, np.ones((5,5),np.uint8), iterations=3)
            labr = Image.fromarray(labnp)
            labr=np.array(labr.resize((168,224),PIL.Image.ANTIALIAS))
            new_lab=[{'x':x,'y':y} for x in range(labr.shape[1]) for y in range(labr.shape[0]) if labr[y,x]==1]
            resized.append(ims[im])
        resized.append(np.array(imr))
        rszlabs.append(labels[im])
        rszlabs.append(new_lab)
    return resized,rszlabs

def translate(im,label):
    if label!='Skip':
        xmin = np.min(np.array([label[p]['x'] for p in range(len(label))]))
        xmax = np.max(np.array([label[p]['x'] for p in range(len(label))]))
        ymin = np.min(np.array([label[p]['y'] for p in range(len(label))]))
        ymax = np.max(np.array([label[p]['y'] for p in range(len(label))]))
    else:
        xmin=50
        xmax=180
        ymin=60
        ymax=200
    if xmax-xmin > ymax-ymin:
        xr=[(im.shape[1]-xmax)/3,2*(im.shape[1]-xmax)/3]
        xl=[2*xmin/3,xmin/3]
        yu=[(im.shape[0]-ymax)/2]
        yd=[ymin/2]
    else:
        xr=[(im.shape[1]-xmax)/2]
        xl=[xmin/2]
        yu=[(im.shape[0]-ymax)/3,2*(im.shape[0]-ymax)/3]
        yd=[2*ymin/3,ymin/3]

    augims = []
    augims.append(copy.deepcopy(im))
    auglabs = []
    auglabs.append(copy.deepcopy(label))

    for t in xr:
        t=int(t)
        translated = np.empty(im.shape)
        for c in range(im.shape[1]):
            if c<im.shape[1]-t:
                translated[:,c+t]=im[:,c]
            else:
                translated[:,c-im.shape[1]+t]=im[:,c]

        augims.append(translated)
        if label!='Skip': auglabs.append([{'x':int(p['x']+t),'y':p['y']} for p in label])
        else: auglabs.append('Skip')


    for t in xl:
        t=int(t)
        translated = np.empty(im.shape)
        for c in range(im.shape[1]):
            if c>=t:
                translated[:,c-t]=im[:,c]
            else:
                translated[:,c+im.shape[1]-t]=im[:,c]
        augims.append(translated)
        if label!='Skip': auglabs.append([{'x':int(p['x']-t),'y':p['y']} for p in label])
        else: auglabs.append('Skip')

    augimz = copy.copy(augims)
    auglabz = copy.copy(auglabs)
    for aim in augims:
        for t in yu:
            t=int(t)
            translated = np.empty(aim.shape)
            for r in range(im.shape[0]):
                if r<aim.shape[0]-t:
                    translated[r+t,:]=aim[r,:]
                else:
                    translated[r-aim.shape[0]+t,:]=aim[r,:]
            augimz.append(translated)
        for t in yd:
            t=int(t)
            translated = np.empty(aim.shape)
            for r in range(im.shape[0]):
                if r>=t:
                    translated[r-t,:]=aim[r,:]
                else:
                    translated[r+aim.shape[0]-t,:]=aim[r,:]
            augimz.append(translated)

    for lab in auglabs:
        for t in yu:
            if label!='Skip': auglabz.append([{'x':p['x'],'y':int(p['y']+t)} for p in lab])
            else: auglabz.append('Skip')
        for t in yd:
            if label!='Skip': auglabz.append([{'x':p['x'],'y':int(p['y']-t)} for p in lab])
            else: auglabz.append('Skip')

    return augimz,auglabz

def invert(img,lab):
    new_ims=[]
    new_labs=[]
    for im in range(img.shape[0]):
        new_ims.append(img[im])
        new_ims.append(np.fliplr(img[im].reshape(-1,3)).reshape(img[im].shape))
        new_labs.append(lab[im])
        new_labs.append(lab[im])
    return np.array(new_ims),np.array(new_labs)

def lab_rot(im,lab):
    a=np.zeros((im.shape[0],im.shape[1]))
    for coord in lab:
        a[coord['y'],coord['x']]=1
    aa = np.rot90(a)
    new_coords=[{'x':x,'y':y} for x in range(aa.shape[1]) for y in range(aa.shape[0]) if aa[y,x]==1]
    return new_coords

def rotate(img, lab):
    rot8d=[]
    rotlabs=[]
    for im in range(img.shape[0]):
        rot8d.append(img[im])
        rotlabs.append(lab[im])
        img90=np.rot90(img[im],axes=(0,1))

        if type(lab[im])!= np.str_: lab90=lab_rot(img[im],lab[im])
        else: lab90=lab[im]
        rot8d.append(img90)
        rotlabs.append(lab90)

        img180=np.rot90(img90,axes=(0,1))
        if type(lab[im])!= np.str_: lab180=lab_rot(img90,lab90)
        else: lab180=lab[im]
        rot8d.append(img180)
        rotlabs.append(lab180)

        img270=np.rot90(img180,axes=(0,1))
        if type(lab[im])!= np.str_: lab270=lab_rot(img180,lab180)
        else: lab270=lab[im]
        rot8d.append(img270)
        rotlabs.append(lab270)

    return rot8d,rotlabs

if __name__=="__main__":

    data_dir = '../data/orig/'

    dataset = []
    for classdir in os.listdir(data_dir):
        if os.path.isdir(data_dir+classdir):
            class_ims = [imageio.imread(data_dir+classdir+f'/{classdir}{im}.png')[:,:,:3] for im in range(len(os.listdir(data_dir+classdir)))] #os.listdir(data_dir+classdir) if im.endswith('png')]
            dataset.extend(class_ims)
    dataset=np.array(dataset)



    labels = pd.read_csv('../data/labelboxout/orig_labeled.csv')
    boxes = [ast.literal_eval(labels.loc[im,'Label'])['coke_bottle'][0]['geometry'] for im in range(dataset.shape[0]) if labels.loc[im,'Label']!="Skip"]
    boxes.extend(['Skip' for im in range(dataset.shape[0]) if labels.loc[im,'Label']=='Skip'])

    resized,rszlabs = resize(dataset,boxes)

    # for img in range(len(resized)):
    # plt.imshow(resized[103])
    # if type(rszlabs[102])!=np.str:
    #     for coord in rszlabs[102]:
    #         plt.scatter(coord['x'],coord['y'])
    # plt.show()


    # print(rszlabs[1])
    # print(len(rszlabs[0]))
    # print(len(rszlabs[1]))

    # print(dataset[0].shape)
    # print(dataset[51].shape)
    final_ims=[]
    final_labs=[]

    for imlab in range(len(resized)):
        imlabaug = np.array(translate(resized[imlab],rszlabs[imlab]))

        imlabaug2=invert(imlabaug[0],imlabaug[1])

        imlabaug3=rotate(imlabaug2[0],imlabaug2[1])
        final_ims.extend(imlabaug3[0])
        final_labs.extend(imlabaug3[1])

    print(len(resized))
    print(len(rszlabs))
    print(len(final_ims))
    print(len(final_labs))
    print(len(resized)/len(final_ims))
    print(len(rszlabs)/len(final_labs))

    for img in range(len(final_ims)):
        imageio.imwrite(f'/Volumes/ElementsExternal/coca_coda/augims/img{img}.png',final_ims[img])
        np.save(f'/Volumes/ElementsExternal/coca_coda/auglabs/lab{img}.txt',final_labs[img])
        # with open('auglabs/lab{img}.txt', 'w') as lab:
        #     lab.write(final_labs[img])



    # for nim in range(len(imlabaug3[0])):
    #     plt.imshow(imlabaug3[0][nim])
    #     if type(imlabaug3[1][nim])!=np.str_:
    #         for coord in imlabaug3[1][nim]:
    #             plt.scatter(coord['x'],coord['y'])
    #     plt.show()
