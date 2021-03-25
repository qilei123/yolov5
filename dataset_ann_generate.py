from pycocotools.coco import COCO
import os

def xywh2cxcywh(bbox):
    return [bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2,bbox[2],bbox[3]]

def anns2gtboxes(gtanns):
    gtboxes = []
    for ann in gtanns:
        box = xywh2cxcywh(ann['bbox'])
        box.insert(0,ann['category_id'])
        #box.insert(0,)
        gtboxes.append(box)
    return gtboxes

def coco2yolov5():
    sets = ['train','test']
    set_name = sets[1] #
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'

    save_folder = "/data1/qilei_chen/DATA/erosive/labels/"+set_name

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    coco_instance = COCO(anns_file)

    coco_imgs = coco_instance.imgs

    for img_id in coco_imgs:
        gtannIds = coco_instance.getAnnIds(imgIds= img_id)
        gtanns = coco_instance.loadAnns(gtannIds)  
        gtboxes = anns2gtboxes(gtanns) 
        img_filename = coco_imgs[img_id]['file_name']

        cp_command = "cp "+"/data1/qilei_chen/DATA/erosive/images/"+img_filename +" /data1/qilei_chen/DATA/erosive/yolov5/images/"+set_name+"/"
        os.system(cp_command)
        label_filename = img_filename[:-3]+"txt"
        label_filedir = os.path.join(save_folder,label_filename)
        label_file = open(label_filedir,'w')
        for gtbox in gtboxes:
            count=0
            for dignum in gtbox:
                if count==0:
                    label_file.write(str(dignum))
                if count==1 or count==3:
                    label_file.write(str(dignum/coco_imgs[img_id]['width']))
                if count==2 or count==4:
                    label_file.write(str(dignum/coco_imgs[img_id]['height']))
                if count==4:
                    label_file.write("\n")
                else:
                    label_file.write(" ")
                count+=1

coco2yolov5()