<div align="center">

# Locality-aware 3D Gaussian Compression <br> for Fast and High-quality Rendering

<div style="font-size:120%">
  <a href="https://seungjooshin.github.io">Seungjoo Shin<a><sup>1</sup> &nbsp;&nbsp;</a>
  <a href="https://jaesik.info">Jaesik Park<a><sup>2</sup> &nbsp;&nbsp;</a>
  <a href="https://www.scho.pe.kr">Sunghyun Cho<a><sup>1</sup> &nbsp;&nbsp;</a>
</div>

<p>
    <sup>1</sup>POSTECH &nbsp;&nbsp;
    <sup>2</sup>Seoul National University &nbsp;&nbsp;
</p>

### <p style="font-size:120%"> ICLR 2025</p>
<a href='https://arxiv.org/abs/2501.05757'><img src='https://img.shields.io/badge/arXiv-2501.05757-b31b1b.svg'></a> &nbsp;&nbsp;
<a href='https://seungjooshin.github.io/LocoGS'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

<img src="./assets/qualitative.png"/>
<br>
</div>

## Installation

This project is built upon the following environment:
- Python 3.10
- CUDA 11.7
- PyTorch 2.0.1

### Conda Environment
Run the following command to build the environment.
```bash
# clone this repo
git clone --recursive https://github.com/seungjooshin/LocoGS.git
cd LocoGS

# create a conda environment
conda create -n locogs python=3.10
conda activate locogs

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# install nerfstudio
cd submodules/nerfstudio
pip install --upgrade pip setuptools
pip install -e .
cd ../../

# install packages
pip install -r requirements.txt
```

### G-PCC
Run the following command to build [TMC13](https://github.com/MPEGGroup/mpeg-pcc-tmc13).
```bash
# build mpeg-pcc-tmc13
cd submodules/mpeg-pcc-tmc13
mkdir build
cd build
cmake ..
make
cd ../../../
```

## Data

We support three datasets for evaluation. Please follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to set up the data.

- [MipNeRF-360](https://jonbarron.info/mipnerf360/) 
- [Tanks & Temples + Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

```shell
├── data
    ├── 360_v2
      ├── bicycle
      ├── bonsai
      ├── ...
    ├── tandt
      ├── ...
    ├── db
      ├── ...
```

## Running

Please refer to the [scripts](scripts) for evaluation.
- Dense initialization: [scripts/run_nerfacto.sh](scripts/run_nerfacto.sh) 
- LocoGS: [scripts/run.sh](scripts/run.sh) 
- LocoGS-Small: [scripts/run_small.sh](scripts/run_small.sh) 
- LocoGS w/o dense initialization: [scripts/run_colmap_pcd.sh](scripts/run_colmap_pcd.sh) 


### Preprocessing

Run the following command to acquire a dense point cloud. 
``` shell
# processing data
ns-process-data images --data data/${DATASET}/${SCENE} \
                       --output-dir data/${DATASET}/${SCENE} \
                       --skip-colmap --skip-image-processing

# train nerfacto
ns-train nerfacto --data data/${DATASET}/${SCENE} \
                  --output-dir output \
                  --timestamp run \
                  --vis tensorboard \
                  --machine.seed 0  \
                  --pipeline.model.camera-optimizer.mode off

# export pointcloud
ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml \
                     --output-dir output/${SCENE}/nerfacto/run \
                     --remove-outliers True \
                     --num-points ${NUM_PTS} \
                     --normal-method open3d \
                     --save-world-frame True
```
The point cloud will be saved as `output/${SCENE}/nerfacto/run/point_cloud.ply`.
- If you get inaccruate results, please re-run COLMAP.

### Training

Run the following command to optimize LocoGS.
``` shell
# LocoGS
python train.py -s data/${DATASET}/${SCENE} -m output/${SCENE} --eval \
                --pcd_path ${PATH_TO_PCD} \
                --lambda_mask 0.004 \
                --hash_size 19

# LocoGS-Small
python train.py -s data/${DATASET}/${SCENE} -m output/${SCENE} --eval \
                --pcd_path ${PATH_TO_PCD} \
                --lambda_mask 0.005 \
                --hash_size 17

# LocoGS w/ COLMAP point cloud
python train.py -s data/${DATASET}/${SCENE} -m output/${SCENE} --eval \
                --lambda_mask 0.001 \
                --hash_size 19
```
- `pcd_path`: path to initial point cloud (COLMAP point cloud by default)
- `lambda_mask`: weight of masking loss (`0.004` by default)
- `hash_size`: size of hash grid, (`19` by default)

### Evaluation
Run the following command to evaluate LocoGS.
```bash
# render LocoGS
python render.py -s data/${DATASET}/${SCENE} -m output/${SCENE}

# compute error metrics on renderings
python metrics.py -m output/${SCENE}
```

### Convert to PLY format
Run the following command to convert LocoGS to PLY format.
```bash
# convert LocoGS to PLY
python convert2ply.py -m output/${SCENE}
```


## Citation

If our work is useful, please consider citing our paper:
```bib
@inproceedings{shin2025localityaware,
  title={Locality-aware Gaussian Compression for Fast and High-quality Rendering},
  author={Seungjoo Shin and Jaesik Park and Sunghyun Cho},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=dHYwfV2KeP}
}
```

## Acknowledgement

Our code is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Compact-3DGS](https://github.com/maincold2/Compact-3DGS). Thanks for their awesome work.
