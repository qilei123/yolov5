export CUDA_VISIBLE_DEVICES=0
python val.py --weights /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s/exp/weights/best.pt --data trans_drone_cat3.yaml --img 1280 --task test