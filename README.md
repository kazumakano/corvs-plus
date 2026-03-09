# CorVS+: Correspondence-Driven Association of Video Trajectories and Sensors for Identity-Aware Person Localization in Warehouses

## News

## Requirements
- Python 3.11

## Installation
Install the `corvs` package and its dependencies.
```sh
git clone https://github.com/kazumakano/corvs-plus.git && cd corvs-plus/
pip install -e ./
```

## Dataset
Coming soon.

## Pre-trained models
Coming soon.

| Ver | # of Params | FLOPs @ 1-min | Test PF | Weights | Hyperparams | Note                |
| --- | ----------- | ------------- | ------- | ------- | ----------- | ------------------- |
| 1   | 106k        | 378M          | 0.761   |         |             | Best model in paper |

## Demo
Infer the correspondence probabilities and reliabilities for a specific pair of a trajectory and sensor measurements.
```sh
python infer.py \
    -d dataset/ \
    -p configs/hparams.yaml \
    -w dataset/model.safetensors \
    -t 7 \
    -s 3 \
    --from '2024-10-03 11:30:00' \
    --to '2024-10-03 12:00:00'
```

| Argument                  | Description                           | Type | Required | Default                     |
| ------------------------- | ------------------------------------- | ---- | -------- | --------------------------- |
| `-d` `--date_path`        | Path to dataset root directory        | str  | Yes      | `dataset/`                  |
| `-p` `--param_path`       | Path to hyperparameter file           | str  | Yes      | `configs/hparams.yaml`      |
| `-w` `--weight_path`      | Path to model weight file             | str  | Yes      | `dataset/model.safetensors` |
| `-t` `--traj_track_id`    | Track ID of trajectory                | int  | Yes      |                             |
| `-s` `--sensor_worker_id` | Worker ID of sensor measurements      | int  | Yes      |                             |
| `--from`                  | Start datetime in Japan Standard Time | str  | No       |                             |
| `--to`                    | End datetime in Japan Standard Time   | str  | No       |                             |

## Contact
Kazuma Kano \
Graduate School of Engineering, Nagoya University \
Email: [kazuma@ucl.nuee.nagoya-u.ac.jp](mailto:kazuma@ucl.nuee.nagoya-u.ac.jp)
