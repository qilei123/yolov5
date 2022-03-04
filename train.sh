git pull
#export CUDA_VISIBLE_DEVICES=0
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1280 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1280_mosaic4
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4

#python train_obb.py --exist-ok --device 1 --img 1280 --batch 1 --epochs 64 --cfg models/yolov5s_obb.yaml --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data/qilei/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb --name exp_1280_mosaic4_debug #--mosaic_n False

python train.py --exist-ok --device 1 --img 1280 --batch 4 --epochs 64 --cfg models/yolov5s_obb.yaml --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data/qilei/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb --name exp_1280_mosaic4_debug #--mosaic_n False