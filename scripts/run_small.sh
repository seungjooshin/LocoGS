#!/bin/bash

SCENE_LIST="bonsai counter kitchen room bicycle garden stump flowers treehill"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        FACTOR=2
    else
        FACTOR=4
    fi
    
    echo python train.py -s data/360_v2/${SCENE} -i images_${FACTOR} --eval --lambda_mask 0.005 --pcd_path output/${SCENE}/nerfacto/run/point_cloud.ply --hash_size 17
    python train.py -s data/360_v2/${SCENE} -i images_${FACTOR} -m output/${SCENE} --eval --lambda_mask 0.005 --pcd_path output/${SCENE}/nerfacto/run/point_cloud.ply --hash_size 17

    echo python render.py -s data/360_v2/${SCENE} -m output/${SCENE}
    python render.py -s data/360_v2/${SCENE} -m output/${SCENE}

    echo python metrics.py -m output/${SCENE}
    python metrics.py -m output/${SCENE}
done

SCENE_LIST="train truck"

for SCENE in $SCENE_LIST;
do
    echo python train.py -s data/tandt/${SCENE} --eval --lambda_mask 0.005 --pcd_path output/${SCENE}/nerfacto/run/point_cloud.ply --hash_size 17
    python train.py -s data/tandt/${SCENE} -m output/${SCENE} --eval --lambda_mask 0.005 --pcd_path output/${SCENE}/nerfacto/run/point_cloud.ply --hash_size 17

    echo python render.py -s data/tandt/${SCENE} -m output/${SCENE}
    python render.py -s data/tandt/${SCENE} -m output/${SCENE}

    echo python metrics.py -m output/${SCENE}
    python metrics.py -m output/${SCENE}
done

SCENE_LIST="drjohnson playroom"

for SCENE in $SCENE_LIST;
do
    echo python train.py -s data/db/${SCENE} --eval --lambda_mask 0.005 --pcd_path output/${SCENE}/nerfacto/run/point_cloud.ply --hash_size 17
    python train.py -s data/db/${SCENE} -m output/${SCENE} --eval --lambda_mask 0.005 --pcd_path output/${SCENE}/nerfacto/run/point_cloud.ply --hash_size 17

    echo python render.py -s data/db/${SCENE} -m output/${SCENE}
    python render.py -s data/db/${SCENE} -m output/${SCENE}

    echo python metrics.py -m output/${SCENE}
    python metrics.py -m output/${SCENE}
done