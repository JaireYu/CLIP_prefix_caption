
#!/bin/bash
# find all configs in configs/
# set your gpu id
gpus=0

["--only_prefix",
 "--data", "./data/coco/oscar_split_ViT-B_32_train.pkl"
 , "--data_val", "./data/coco/oscar_split_ViT-B_32_val.pkl", 
 "--out_dir", "./coco_train/", 
 "--mapping_type", "transformer", 
 "--num_layers", "8", 
 "--prefix_length", "40", 
 "--prefix_length_clip", "40", 
 "--normalize_prefix"]

method_id=BASE_LINE

seed=42
do
    echo "training using the $seed seed\n"
    mkdir ./outputs/$model\_$method\_test_$seed\_gpu_$gpus\_XYseperate_Uni_0.1_SPARSE_SOFT_0.00001
    output_dir=./outputs/$model\_$method\_test_$seed\_gpu_$gpus\_XYseperate_Uni_0.1_SPARSE_SOFT_0.00001

    CUDA_VISIBLE_DEVICES=$gpus nohup python \
    train.py --config-file $config_file --seed $seed OUTPUT_DIR $output_dir SOLVER.MAX_EPOCH 20 SOLVER.LR 0.0001 SOLVER.BATCH_SIZE 64 TEST.BATCH_SIZE 64 MODEL.ARCHITECTURE $method \
    MODEL.TCN.FEATATTN.QUERYTYPE XYseperate MODEL.TCN.FEATATTN.POSINIT Uniform MODEL.TCN.FEATATTN.ALPHA -0.1 MODEL.TCN.FEATATTN.BETA 0.1 MODEL.TCN.FEATATTN.SIM_CONF 0.00001 MODEL.TCN.FEATATTN.SPARSE_GT SOFT > $output_dir/nohup.out &
done