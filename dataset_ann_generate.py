from pycocotools.coco import COCO
import os

def xywh2cxcywh(bbox):
    return [bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2,bbox[2],bbox[3]]

def anns2gtboxes(gtanns):
    gtboxes = []
    for ann in gtanns:
        box = xywh2cxcywh(ann['bbox'])
        print(ann)
        #box.insert(0,)
        gtboxes.append(box)
    return gtboxes

def coco2yolov5():
    sets = ['train','test']
    set_name = sets[0] #
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'

    save_folder = "/data1/qilei_chen/DATA/erosive/labels/"+set_name

    coco_instance = COCO(anns_file)

    coco_imgs = coco_instance.imgs

    for img_id in coco_imgs:
        gtannIds = coco_instance.getAnnIds(imgIds= img_id)
        gtanns = coco_instance.loadAnns(gtannIds)  
        gtboxes = anns2gtboxes(gtanns) 
