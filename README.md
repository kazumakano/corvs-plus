<h1 align="center">CorVS+: Correspondence-Driven Association of Video Trajectories and Sensors for Identity-Aware Person Localization in Warehouses</h1>

<p align="center">
  Kazuma Kano · Yuki Mori · Shin Katayama · Kenta Urano · Takuro Yonezawa · Nobuo Kawaguchi <br>
  Graduate School of Engineering, Nagoya University
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.17745683">
    <img alt="dataset badge" src="https://img.shields.io/badge/Zenodo-Dataset-2f6fa7?logo=zenodo">
  </a>
</p>

![identification process](https://github.com/user-attachments/assets/17c8039a-89ea-46ac-8757-867dddc49afd)

## 📢 News

## 📋 Requirements
- Python 3.11

## ⚡ Installation
Install the `corvs` package and its dependencies.

```sh
git clone https://github.com/kazumakano/corvs-plus.git && cd corvs-plus/
pip install -e ./
```

## 📦 Dataset
Coming soon.

![labeled video](https://github.com/user-attachments/assets/6784a7a2-41c1-4475-b133-9b5a913e6431)

## 🧠 Pre-trained models
Coming soon.

| Ver | # of Params | FLOPs @ W=600 | Test PF | Weights | Hyperparams             | Note                |
| --- | ----------- | ------------- | ------- | ------- | ----------------------- | ------------------- |
| 1   | 106k        | 378M          | 0.761   |         | [hparams.yaml][param 1] | Best model in paper |

[param 1]: https://onedrive.live.com/personal/5ba997f23749e33e/_layouts/15/download.aspx?UniqueId=814f01d0%2Df0c0%2D4be6%2D8155%2De2a5cfb9473b

## 🚀 Demo
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
<details>
  <summary>Command line arguments</summary>
  <p>
    <table>
      <thead>
        <tr>
          <th>Argument</th>
          <th>Description</th>
          <th>Type</th>
          <th>Required</th>
          <th>Default</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>-d</code> <code>--data_path</code></td>
          <td>Path to dataset root directory</td>
          <td>str</td>
          <td></td>
          <td><code>dataset/</code></td>
        </tr>
        <tr>
          <td><code>-p</code> <code>--param_path</code></td>
          <td>Path to hyperparameter file</td>
          <td>str</td>
          <td></td>
          <td><code>configs/hparams.yaml</code></td>
        </tr>
        <tr>
          <td><code>-w</code> <code>--weight_path</code></td>
          <td>Path to model weight file</td>
          <td>str</td>
          <td></td>
          <td><code>dataset/model.safetensors</code></td>
        </tr>
        <tr>
          <td><code>-t</code> <code>--traj_track_id</code></td>
          <td>Track ID of trajectory</td>
          <td>int</td>
          <td>Yes</td>
          <td></td>
        </tr>
        <tr>
          <td><code>-s</code> <code>--sensor_worker_id</code></td>
          <td>Worker ID of sensor measurements</td>
          <td>int</td>
          <td>Yes</td>
          <td></td>
        </tr>
        <tr>
          <td><code>--from</code></td>
          <td>Start datetime in Japan Standard Time</td>
          <td>str</td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td><code>--to</code></td>
          <td>End datetime in Japan Standard Time</td>
          <td>str</td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <td><code>--devices</code></td>
          <td>Computation device indices</td>
          <td>list[int]</td>
          <td></td>
          <td>[0]</td>
        </tr>
      </tbody>
    </table>
  </p>
</details>

## 📜 Citation

## 📬 Contact
Kazuma Kano \
Graduate School of Engineering, Nagoya University \
Email: [kazuma@ucl.nuee.nagoya-u.ac.jp](mailto:kazuma@ucl.nuee.nagoya-u.ac.jp)
