git pull
export CUDA_VISIBLE_DEVICES=0
root_dir1=/data/qilei
root_dir2=/data3/qilei_chen
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1280 --batch 1 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project $root_dir1/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1280_mosaic4 --exist-ok
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4

python train_obb.py --exist-ok \
                    --device 0 \
                    --img 1280 \
                    --batch 32 \
                    --epochs 64 \
                    --hyp data/hyps/hyp.scratch_td.yaml \
                    --cfg models/yolov5s_obb.yaml \
                    --data trans_drone_cat3.yaml \
                    --weights yolov5s.pt \
                    --project $root_dir1/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb \
                    --name exp_1280_mosaic4_debug \
                    --rect #--mosaic_n False

#python train.py --exist-ok --device 1 --img 1280 --batch 4 --epochs 64 --cfg models/yolov5s.yaml --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data/qilei/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb --name exp_1280_mosaic4_debug #--mosaic_n False