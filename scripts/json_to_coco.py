import numpy as np
import json
import labelbox.exporters.coco_exporter as lb2coco
import subprocess

def bash_cmd(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)#, stderr=subprocess.STDOUT)
    output, error = process.communicate()
    with open('preproc_log.txt','a') as lt:
        lt.write(f'Command: {cmd}\nOutput: {[output,error]}\n')
    return

for jsonf in ['test_skip.json']: ## # ,'validation_skip.json','test_skip.json']'train_skip.json'
    print(jsonf)
    with open(f'../data/labelboxout/{jsonf}','r') as json_f:
        fjson = json.load(json_f)
    for im_dict in range(len(fjson)):
        new_lab = np.load(f'/Volumes/ElementsExternal/coca_coda/{jsonf.split("_")[0]}/labels/lab{fjson[im_dict]["External ID"].split(".")[0].split("img")[1]}.txt.npy')
        if str(new_lab) != "Skip":
            fjson[im_dict]['Label']={"coke_bottle":[{"geometry":list(new_lab)}]}
        else:
            fjson[im_dict]['Label']=str(new_lab)

    print(type(fjson))
    with open(f'../data/labelboxout/{jsonf.split("_")[0]}_labels.json','w') as json_lab:
        json.dump(fjson, json_lab)

    #
    # labeled_data = f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels.json'
    # coco_output  = f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels_coco.json'
    #
    # lb2coco.from_json(labeled_data, coco_output, label_format='XY')

    end=len(fjson)
    for step in range(0,len(fjson),250):
        print(step)
        with open(f'../data/labelboxout/temp_step.json','w') as temp:
            json.dump(fjson[step:step+250],temp)
        labeled_data = f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/temp_step.json'
        coco_output  = f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/temp_step_coco.json'

        lb2coco.from_json(labeled_data, coco_output, label_format='XY')
        with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/temp_step_coco.json','r') as tempr:
            ts = json.load(tempr)
        if step == 0:
            with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels_coco.json','w') as coco:
                json.dump(ts,coco)
        else:
            with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels_coco.json','r') as cocor:
                main = json.load(cocor)
            # main.append(ts)
            main['images'].extend(ts['images'])
            main['annotations'].extend(ts['annotations'])
            with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels_coco.json','w') as coco:
                json.dump(main,coco)
        last = step
        print('done with step')
    with open(f'../data/labelboxout/temp_step.json','w') as temp:
        json.dump(fjson[last:],temp)
    labeled_data = f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/temp_step.json'
    coco_output  = f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/temp_step_coco.json'

    lb2coco.from_json(labeled_data, coco_output, label_format='XY')
    with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/temp_step_coco.json','r') as tempr:
        ts = json.load(tempr)
    with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels_coco.json','r') as coco:
        main = json.load(coco)
    # main.append(ts)
    main['images'].extend(ts['images'])
    main['annotations'].extend(ts['annotations'])
    with open(f'/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/{jsonf.split("_")[0]}_labels_coco.json','w') as coco:
            json.dump(main,coco)
    print('made it')

# bash_cmd(f'python ../submodules/TransferLearningToolchain/scripts/create_coco_tf_record.py --train_annotations_file=/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/train_labels_coco.json --val_annotations_file=/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/validation_labels_coco.json --testdev_annotations_file=/Users/xtian/Documents/GitHub/coca_coda/data/labelboxout/test_labels_coco.json   --train_image_dir=/Volumes/ElementsExternal/coca_coda/train/ims --val_image_dir=/Volumes/ElementsExternal/coca_coda/validation/ims --test_image_dir=/Volumes/ElementsExternal/coca_coda/test/ims --output_dir=../data/tfrecords')
