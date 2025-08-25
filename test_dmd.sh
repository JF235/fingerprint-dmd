#!/bin/bash

# 1. Go to the enhanced dataset directory
cd datasets/test_data

# 2. Convert image formats
# /sd258_fnet_gabor/enhanced
python convert_img.py \
    --source_dir /localStorage/data/datasets/SD258/enhanced/ \
    --target_dir . \
    --dataset_name sd258_fnet_gabor \
    --query_regex '.+?-(\d+).+?\.png' \
    --query_value 00 \
    --debug

# a. Source directory of images
# b. The target directory of test data (in this case, the test_data)
# c. The directory where the converted images will be saved
# d. The query regex to match queries
#   sd258_126_11-01_template_good.png -> .+?-(\d+).+?\.png
# 
# e. The query value to match indicated by the group, parenthesis
#   The query is the latent, indicated by 00

# 3. Convert mnt formats
python convert_mnt.py \
    --source_dir /localStorage/data/datasets/SD258/minutiae \
    --target_dir . \
    --dataset_name sd258_fnet_gabor \
    --query_regex '.+?-(\d+).+?\.txt' \
    --query_value 00 \
    --debug

# 4. Create pairs
python generate_pairs.py \
    --dataset_dir sd258_fnet_gabor/ \
    --subject_regex "sd258_(\d{3})_\d+-\d{2}_.*\.mnt" \
    --debug

# Go back to root
cd ../..

# 5. Extract patches
python dump_dataset_mnteval.py --prefix datasets/test_data --dataset_name sd258_fnet_gabor --img_type png 

# 6. Perform eval
python evaluate_mnt.py -d sd258_fnet_gabor -m DMD++ -sn -e                                               