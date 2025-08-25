"""
This file (dump_dataset_mnteval.py) is designed for:
    dump dataset for DMD evaluating
Copyright (c) 2024, Zhiyu Pan. All rights reserved.
"""
import os
import os.path as osp
import pickle
import random
import numpy as np
import argparse
from fptools import fp_verifinger
area_thresh = 40000

def create_datalist(prefix, dataname, img_type='bmp'):
    # NIST4
    img_lst = []
    anchor_2d = []
    mnt_folder = osp.join(prefix, f'{dataname}/mnt') # using the extracted by fingernet
    mnt_gallery_folder = osp.join(mnt_folder, 'gallery')
    mnt_query_folder = osp.join(mnt_folder, 'query')
    mnt_gallery_files = os.listdir(mnt_gallery_folder)
    mnt_query_files = os.listdir(mnt_query_folder)
    for mnt_f in mnt_gallery_files: 
        # This can be loaded by other functions for specific minutia files; just make sure the first three columns of mnts are (x, y, theta).
        mnts = fp_verifinger.load_minutiae(osp.join(mnt_gallery_folder, mnt_f)) # [N, 3] (x, y, theta), theta in clockwise
        for mnt_ in mnts: # one mnt per sample
            img_lst.append(osp.join(f"{dataname}", "image", 'gallery', mnt_f.split('.')[0] + f".{img_type}")) 
            anchor_2d.append(mnt_)

    for mnt_f in mnt_query_files:
        try:
            mnts = fp_verifinger.load_minutiae(osp.join(mnt_query_folder, mnt_f))[:, :3]  # [N, 3] (x, y, theta), theta in clockwise
        except IndexError as e:
            print(mnt_f,e)
        for mnt_ in mnts:
            img_lst.append(osp.join(f"{dataname}", "image", 'query', mnt_f.split('.')[0] + f".{img_type}"))
            anchor_2d.append(mnt_)

    data_lst = {"img": img_lst,  "pose_2d": anchor_2d}
    print(f'{dataname} total {len(img_lst)} samples')
    return data_lst

if __name__ == "__main__":
    random.seed(1016)
    np.random.seed(1016)
    parser = argparse.ArgumentParser("Evaluation for DMD")
    parser.add_argument("--prefix", type=str, default="/path/to/TEST_DATA")
    parser.add_argument("--dataset_name", type=str, default="sd258")
    parser.add_argument("--img_type", type=str, default="png")
    args = parser.parse_args()
    # # the NIST series dataset
    datasets = [args.dataset_name] #
    img_types = [args.img_type] #
    for dataset, img_type in zip(datasets, img_types):
        datalist = create_datalist(args.prefix, dataset, img_type)
        datalist = [dict(zip(datalist, v)) for v in zip(*datalist.values())]
        save_file = f'./datasets/{dataset}.pkl'
        # save the data
        with open(save_file, "wb") as fp:
            pickle.dump(datalist, fp)


