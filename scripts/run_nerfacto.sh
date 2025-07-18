#!/bin/bash

NUM_PTS=1_000_000


SCENE_LIST="bonsai counter kitchen room bicycle garden stump flowers treehill"

for SCENE in $SCENE_LIST;
do
    # processing data
    echo ns-process-data images --data data/360_v2/${SCENE} --output-dir data/360_v2/${SCENE} --skip-colmap --skip-image-processing
    ns-process-data images --data data/360_v2/${SCENE} --output-dir data/360_v2/${SCENE} --skip-colmap --skip-image-processing
    
    # train nerfacto
    echo ns-train nerfacto --data data/360_v2/${SCENE} --output-dir output --timestamp run --vis tensorboard --machine.seed 0 --pipeline.model.camera-optimizer.mode off
    ns-train nerfacto --data data/360_v2/${SCENE} --output-dir output --timestamp run --vis tensorboard --machine.seed 0 --pipeline.model.camera-optimizer.mode off

    # export pointcloud
    echo ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml --output-dir output/${SCENE}/nerfacto/run --remove-outliers True --num-points ${NUM_PTS} --normal-method open3d --save-world-frame True 
    ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml --output-dir output/${SCENE}/nerfacto/run --remove-outliers True --num-points ${NUM_PTS} --normal-method open3d --save-world-frame True
done

SCENE_LIST="train truck"

for SCENE in $SCENE_LIST;
do
    # processing data
    echo ns-process-data images --data data/tandt/${SCENE} --output-dir data/tandt/${SCENE} --skip-colmap --skip-image-processing
    ns-process-data images --data data/tandt/${SCENE} --output-dir data/tandt/${SCENE} --skip-colmap --skip-image-processing
    
    # train nerfacto
    echo ns-train nerfacto --data data/tandt/${SCENE} --output-dir output --timestamp run --vis tensorboard --machine.seed 0 --pipeline.model.camera-optimizer.mode off
    ns-train nerfacto --data data/tandt/${SCENE} --output-dir output --timestamp run --vis tensorboard --machine.seed 0 --pipeline.model.camera-optimizer.mode off

    # export pointcloud
    echo ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml --output-dir output/${SCENE}/nerfacto/run --remove-outliers True --num-points ${NUM_PTS} --normal-method open3d --save-world-frame True 
    ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml --output-dir output/${SCENE}/nerfacto/run --remove-outliers True --num-points ${NUM_PTS} --normal-method open3d --save-world-frame True
done

SCENE_LIST="drjohnson playroom"

for SCENE in $SCENE_LIST;
do
    # processing data
    echo ns-process-data images --data data/db/${SCENE} --output-dir data/db/${SCENE} --skip-colmap --skip-image-processing
    ns-process-data images --data data/db/${SCENE} --output-dir data/db/${SCENE} --skip-colmap --skip-image-processing
    
    # train nerfacto
    echo ns-train nerfacto --data data/db/${SCENE} --output-dir output --timestamp run --vis tensorboard --machine.seed 0 --pipeline.model.camera-optimizer.mode off
    ns-train nerfacto --data data/db/${SCENE} --output-dir output --timestamp run --vis tensorboard --machine.seed 0 --pipeline.model.camera-optimizer.mode off

    # export pointcloud
    echo ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml --output-dir output/${SCENE}/nerfacto/run --remove-outliers True --num-points ${NUM_PTS} --normal-method open3d --save-world-frame True 
    ns-export pointcloud --load-config output/${SCENE}/nerfacto/run/config.yml --output-dir output/${SCENE}/nerfacto/run --remove-outliers True --num-points ${NUM_PTS} --normal-method open3d --save-world-frame True
done