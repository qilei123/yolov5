#git pull
export CUDA_VISIBLE_DEVICES=0,1
root_dir1=/data/qilei
root_dir2=/data3/qilei_chen
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
python train_obb_v1.py --img 1280 --batch 32 --epochs 64 --data trans_drone_cat5.yaml --weights yolov5s.pt --project $root_dir1/work_model_dirs/yolov5/trans_drone_cat5/yolov5s --name exp_1280_mosaic4_obb_v1 --exist-ok --device 1 --hyp data/hyps/hyp.scratch_td.yaml
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4
#python train.py --img 1920 --batch 8 --epochs 64 --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data3/qilei_chen/work_model_dirs/yolov5/trans_drone_cat3/yolov5s --name exp_1920_mosaic4

#python train_obb.py --exist-ok \
#                    --device 0 \
#                    --img 1280 \
#                    --batch 16 \
#                    --epochs 128 \
#                    --hyp data/hyps/hyp.scratch_td.yaml \
#                    --cfg models/yolov5s_obb.yaml \
#                    --data trans_drone_cat3.yaml \
#                    --weights yolov5s.pt \
#                    --project $root_dir1/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb \
#                    --name exp_1280_mosaic4_debug1 \
#                    --rect #--mosaic_n false

#python train.py --exist-ok --device 1 --img 1280 --batch 4 --epochs 64 --cfg models/yolov5s.yaml --data trans_drone_cat3.yaml --weights yolov5s.pt --project /data/qilei/work_dir_models_results/yolov5/trans_drone_cat3/yolov5s_obb --name exp_1280_mosaic4_debug #--mosaic_n False