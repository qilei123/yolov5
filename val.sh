export CUDA_VISIBLE_DEVICES=1
git pull
#python val.py --weights /data/qilei/work_model_dirs/yolov5/trans_drone_cat3/yolov5s/exp_1280_mosaic4/weights/best.pt --data trans_drone_cat3.yaml --img 1280 --task test --batch-size 1

python val_obb.py --weights /data/qilei/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb/exp_1280_mosaic4_debug/weights/last.pt --data trans_drone_cat3.yaml --img 1280 --task test --batch-size 32