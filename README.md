# task-vector-adv

## Requirements
Requirements are provided in ``requirements.txt``.

## Model Training and Testing
Train standard models.
```
python train_corruption.py \
    --pretrained --lr 0.01 \
    --dataset <dataset> --arch <arch> \
    --data <training data> --std_data <standard data> \
    --pretrained_path <pre-trained model path> --save_dir <path for checkpoint>
```

Train robust models.
```
python train_corruption.py \
    --pretrained --lr 0.01 \
    --dataset <dataset> --arch <arch> --corruption <corruption type> \
    --data <training data> --std_data <standard data> \
    --pretrained_path <pre-trained model path> --save_dir <path for checkpoint>
```

Model Patching.
```
python model_patching.py --keep_num <num of layers used> --dataset <dataset> --arch <arch> \
        --corruption <corruption type> --serverity <severity level> --data <std testing data> --corruption_data <corrupted testing data>
        --corruption_model_root <root to store all robust models> \
        --base_model <root to base model> --pretrained <root to pretrained model --save_log <path to save log>
```