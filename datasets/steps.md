# A. Adapting your data

All below at `datasets/test_data`

1. Move the images to correct place

```bash
python convert_img.py \
    --source_dir /fast8TB/jcontreras/data/datasets/SD258/orig/ \
    --target_dir . \
    --dataset_name sd258 \
    --query_regex 'sd258_\d+_\d+-(\d{2}).*\.png' \
    --query_value 00 \
    --debug
```

or

```
python create_test.py --source_dir /localStorage/data/datasets/FVC2002Db1A/orig/ --target_dir . --dataset_name fvc2002db1a --query_regex '\d+_(\d).png' --query_value 1 --debug
```

2. Move the mnt and convert to correct place

2.1 Extrat mnt if needed

```
fingernet infer SD258/orig SD258/fingernet_out --gpus "[1,2]" -b 32 --recursive --compile --cpu-workers 8
```

2.2 Convert to format

```bash
python convert_mnt.py \
    --source_dir /localStorage/data/datasets/SD258/minutiae_verifinger/ \
    --target_dir test_data \
    --dataset_name sd258 \
    --query_regex 'sd258_\d+_\d+-(\d{2})_' \
    --query_value 00 \
    --debug
```

```bash
python convert_mnt.py \
    --source_dir /localStorage/data/datasets/FVC2002Db1A/minutiae/ \
    --target_dir . \
    --dataset_name fvc2002db1a \
    --query_regex '\d+_(\d)\.txt' \
    --query_value 1 \
    --debug
```

IMPORTANT: Minutiae Format !!!


| Property | Value | Evidence |
|----------|-------|----------|
| **Units** | **DEGREES** | `np.deg2rad()` / `np.rad2deg()` usado em todo o código |
| **Rotation Direction** | **COUNTER-CLOCKWISE** | Segue convenção matemática padrão |
| **Zero Reference** | **RIGHT (0° = East)** | `cos(0°)=1, sin(0°)=0` aponta para direita |
| **Range** | **[-180, 180] or [0, 360]** | Normalizado com módulo 360 |

3. Generate genuine pairs

$ python generate_pairs.py --dataset_dir fvc2002db1a --subject_regex '(\d+)_\d.mnt'

# B. Pickle Setup

Now change directories to root

Generate Pickle file

$ python dump_dataset_mnteval.py --prefix datasets/test_data --dataset_name fvc2002db1a --img_type png

$ python dump_dataset_mnteval.py --prefix datasets/test_data --dataset_name sd258 --img_type png      

Be careful with missing minutiae

# C. Evaluating

$ python evaluate_mnt.py -d sd258 -m DMD++ -sn -e

# D. Troubleshooting

- Dependencias extras: easy dict, torch linear assignemtn, opencv contrib (parar createThinPlateSplineShapeTransformer)