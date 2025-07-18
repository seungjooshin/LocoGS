import os
import numpy as np
import torch
import subprocess
from plyfile import PlyData, PlyElement
from utils.general_utils import mortonEncode

TM3_PATH = 'submodules/mpeg-pcc-tmc13/build/tmc3/tmc3'

def encode_xyz(xyz, path, show=True):
    ply_path, bin_path = os.path.join(path, 'xyz.ply'), os.path.join(path, 'xyz.bin')

    neg = xyz < 0
    xyz[neg] *= -1
    xyz = xyz.astype(np.float16).view(np.int16).astype(np.float32)
    xyz[neg] *= -1

    elements = np.empty(xyz.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    elements[:] = list(map(tuple, xyz))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)

    gpcc_encode(ply_path=ply_path, bin_path=bin_path)
    os.remove(ply_path)

def decode_xyz(path, show=True):
    ply_path, bin_path = os.path.join(path, 'xyz.ply'), os.path.join(path, 'xyz.bin')
    gpcc_decode(ply_path=ply_path, bin_path=bin_path)
    with open(ply_path, 'rb') as f:
        plydata = PlyData.read(f)
    
    xyz = np.stack([plydata['vertex'].data[axis] for axis in ('x', 'y', 'z')], axis=-1)
    neg = xyz < 0
    xyz[neg] *= -1
    xyz = xyz.astype(np.int16).view(np.float16).astype(np.float16)
    xyz[neg] *= -1

    xyz_ = xyz.copy().astype(np.float32)
    xyz_q = (xyz_ - xyz_.min(0)) / (xyz_.max(0) - xyz_.min(0)) * (2 ** 21 - 1)
    order = mortonEncode(torch.tensor(xyz_q).long()).sort().indices
    
    xyz = xyz[order]
    
    os.remove(ply_path)

    return xyz.astype(np.float32)

# Modified implementation from PCGCv1:
# https://github.com/NJUVISION/PCGCv1/blob/6aa772a4d00b74ee90e944e197ab9bd665762815/myutils/gpcc_wrapper.py
def gpcc_encode(ply_path, bin_path, show=True):
    cmd = f'{TM3_PATH} \
        --mode=0 \
        --mergeDuplicatedPoints=0 \
        --positionQuantizationScale=1 \
        --neighbourAvailBoundaryLog2=8 \
        --intra_pred_max_node_size_log2=6 \
        --disableAttributeCoding=1 \
        --uncompressedDataPath={ply_path} \
        --compressedStreamPath={bin_path} \
        --transformType=0 \
        --numberOfNearestNeighborsInPrediction=3 \
        --levelOfDetailCount=16 \
        --positionQuantizationScaleAdjustsDist2=1 \
        --dist2=8'

    print("GPCC encoding...")
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    if show:
        line = subp.stdout.readline()
        while line:
            print(line)
            line = subp.stdout.readline()

# Modified implementation from PCGCv1:
# https://github.com/NJUVISION/PCGCv1/blob/6aa772a4d00b74ee90e944e197ab9bd665762815/myutils/gpcc_wrapper.py
def gpcc_decode(ply_path, bin_path, show=True):
    cmd = f'{TM3_PATH} \
        --mode=1 \
        --transformType=0 \
        --compressedStreamPath={bin_path} \
        --reconstructedDataPath={ply_path}'
    
    print("GPCC decoding...")
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    if show:
        line = subp.stdout.readline()
        while line:
            print(line)
            line = subp.stdout.readline()
