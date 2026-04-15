# CorVS+: Correspondence-Driven Association of Video Trajectories and Sensors for Identity-Aware Person Localization in Warehouses
![identification process](https://github.com/user-attachments/assets/a8250375-e635-4bf3-a5eb-a5c666a005c7)

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

![labeled trajectories](https://github.com/user-attachments/assets/6784a7a2-41c1-4475-b133-9b5a913e6431)

## Pre-trained models
Coming soon.

| Ver | # of Params | FLOPs @ W=600 | Test PF | Weights | Hyperparams             | Note                |
| --- | ----------- | ------------- | ------- | ------- | ----------------------- | ------------------- |
| 1   | 106k        | 378M          | 0.761   |         | [hparams.yaml][param 1] | Best model in paper |

[param 1]: https://onedrive.live.com/personal/5ba997f23749e33e/_layouts/15/download.aspx?UniqueId=814f01d0%2Df0c0%2D4be6%2D8155%2De2a5cfb9473b

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
  --to '2024-10-03 12:00:00' \
  --devices 0
```

| Argument                  | Description                           | Type      | Required | Default                     |
| ------------------------- | ------------------------------------- | --------- | -------- | --------------------------- |
| `-d` `--data_path`        | Path to dataset root directory        | str       |          | `dataset/`                  |
| `-p` `--param_path`       | Path to hyperparameter file           | str       |          | `configs/hparams.yaml`      |
| `-w` `--weight_path`      | Path to model weight file             | str       |          | `dataset/model.safetensors` |
| `-t` `--traj_track_id`    | Track ID of trajectory                | int       | Yes      |                             |
| `-s` `--sensor_worker_id` | Worker ID of sensor measurements      | int       | Yes      |                             |
| `--from`                  | Start datetime in Japan Standard Time | str       |          |                             |
| `--to`                    | End datetime in Japan Standard Time   | str       |          |                             |
| `--devices`               | Computation device indices            | list[int] |          | `[0]`                       |

## Contact
Kazuma Kano \
Graduate School of Engineering, Nagoya University \
Email: [kazuma@ucl.nuee.nagoya-u.ac.jp](mailto:kazuma@ucl.nuee.nagoya-u.ac.jp)
