#!/bin/bash

SCENE_LIST="bonsai counter kitchen room bicycle garden stump flowers treehill"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        FACTOR=2
    else
        FACTOR=4
    fi

    echo python train.py -s data/360_v2/${SCENE} -i images_${FACTOR} -m output/${SCENE} --lambda_mask 0.001 --eval
    python train.py -s data/360_v2/${SCENE} -i images_${FACTOR} -m output/${SCENE} --lambda_mask 0.001 --eval

    echo python render.py -s data/360_v2/${SCENE} -m output/${SCENE}
    python render.py -s data/360_v2/${SCENE} -m output/${SCENE}

    echo python metrics.py -m output/${SCENE}
    python metrics.py -m output/${SCENE}
done

SCENE_LIST="train truck"

for SCENE in $SCENE_LIST;
do
    echo python train.py -s data/tandt/${SCENE} --lambda_mask 0.001 --eval
    python train.py -s data/tandt/${SCENE} -m output/${SCENE} --lambda_mask 0.001 --eval

    echo python render.py -s data/tandt/${SCENE} -m output/${SCENE}
    python render.py -s data/tandt/${SCENE} -m output/${SCENE}

    echo python metrics.py -m output/${SCENE}
    python metrics.py -m output/${SCENE}
done

SCENE_LIST="drjohnson playroom"

for SCENE in $SCENE_LIST;
do
    echo python train.py -s data/db/${SCENE} --lambda_mask 0.001 --eval
    python train.py -s data/db/${SCENE} -m output/${SCENE} --lambda_mask 0.001 --eval

    echo python render.py -s data/db/${SCENE} -m output/${SCENE}
    python render.py -s data/db/${SCENE} -m output/${SCENE}

    echo python metrics.py -m output/${SCENE}
    python metrics.py -m output/${SCENE}
done