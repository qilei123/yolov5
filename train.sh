git pull
export CUDA_VISIBLE_DEVICES=0
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1280 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1280_mosaic4
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4

python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5x.pt --project /data/qilei/work_dir_models_results/yolov5/trans_drone_cat3/yolov5x --name exp_1920_mosaic4 #--mosaic_n False