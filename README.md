<h1 align="center">CorVS+: Correspondence-Driven Association of Video Trajectories and Sensors for Identity-Aware Person Localization in Warehouses</h1>

<p align="center">
  Kazuma Kano · Yuki Mori · Shin Katayama · Kenta Urano · Takuro Yonezawa · Nobuo Kawaguchi <br>
  Graduate School of Engineering, Nagoya University
</p>

<p align="center">
  <a href="https://doi.org/10.48550/arXiv.2510.26369"><img alt="Paper badge" src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
  &nbsp;
  <a href="https://doi.org/10.5281/zenodo.17745683"><img alt="Dataset badge" src="https://img.shields.io/badge/Zenodo-Dataset-blue?logo=zenodo"></a>
</p>

![identification process](https://github.com/user-attachments/assets/17c8039a-89ea-46ac-8757-867dddc49afd)

## 📢 News
- **Jul 27, 2026** \
  Paper and dataset have been released!

## 📋 Requirements
- **Python** \
  3.11 (Recommended) | 3.12 | 3.13

## ⚡ Installation
Install the `corvs` package and its dependencies.

```sh
git clone https://github.com/kazumakano/corvs-plus.git && cd corvs-plus/
pip install -e ./
```

## 📦 Dataset
The dataset is available in a [Zenodo repository](https://doi.org/10.5281/zenodo.17745683).
It contains visual tracking trajectories and wearable sensor measurements collected in a logistics warehouse.

![labeled video](https://github.com/user-attachments/assets/6784a7a2-41c1-4475-b133-9b5a913e6431)

Download and extract the data into the `data/` directory.
```sh
scripts/download_data.sh
```

## 🧠 Pre-trained models
Pre-trained model weights are also available in the [Zenodo repository](https://doi.org/10.5281/zenodo.17745683) or via the links below.

| Ver | # of Params | FLOPs @ W=600 | Test PF | Hyperparams            | Weights               | Note |
| --- | ----------- | ------------- | ------- | ---------------------- | --------------------- | ---- |
| 1   | 106k        | 378M          | 0.761   | [hparams_v1][param v1] | [model_v1][weight v1] |      |

[param v1]: https://drive.google.com/file/d/1TlVHEOuue-NeltztdeMwXkQ0WkYTteHD/view
[weight v1]: https://drive.google.com/file/d/1ZTUIQ-0S3dzJkoasVq-TDJaBLwNYj7eJ/view

Download the weights into the `models/` directory.
```sh
scripts/download_model.sh
```

## 🚀 Demo
Infer the correspondence probabilities and reliabilities for a specific pair of a trajectory and sensor measurements.

```sh
python demo.py \
  -d data/ \
  -p configs/hparams.yaml \
  -w models/model.safetensors \
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
          <td><code>data/</code></td>
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
          <td><code>models/model.safetensors</code></td>
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
          <td><code>[0]</code></td>
        </tr>
      </tbody>
    </table>
  </p>
</details>

## 📜 Citation
```bib
@misc{corvs-plus,
  title={CorVS+: Correspondence-Driven Association of Video Trajectories and Sensors for Identity-Aware Person Localization in Warehouses},
  author={Kazuma Kano and Yuki Mori and Shin Katayama and Kenta Urano and Takuro Yonezawa and Nobuo Kawaguchi},
  year={2026},
  eprint={2510.26369},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.26369}
}
```

## 📬 Contact
Kazuma Kano \
Graduate School of Engineering, Nagoya University \
Email: [kazuma@ucl.nuee.nagoya-u.ac.jp](mailto:kazuma@ucl.nuee.nagoya-u.ac.jp)
