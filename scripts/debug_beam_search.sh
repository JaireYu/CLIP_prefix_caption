
#!/bin/bash
# find all configs in configs/
# set your gpu id
gpus=4

method_id=BASE_LINE_only_prefix_beam_search # not using beam search

seed=42

echo "training using the $seed seed\n"

output_dir=./data/output/coco/$method_id

mkdir $output_dir

CUDA_VISIBLE_DEVICES=$gpus nohup python -u \
train.py --only_prefix --seed $seed --data ./data/coco/oscar_split_ViT-B_32_train.pkl --data_val ./data/coco/oscar_split_ViT-B_32_val.pkl \
--out_dir $output_dir --mapping_type transformer --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --normalize_prefix --beam_search > $output_dir/nohup.out &
