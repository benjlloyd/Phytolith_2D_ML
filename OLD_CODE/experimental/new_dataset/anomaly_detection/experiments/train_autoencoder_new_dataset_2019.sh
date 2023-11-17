#!/usr/bin/env bash

name=$1
tag=$2


DIR="$( cd "$( dirname -- "$0" )" && pwd )"
export PYTHONPATH="${DIR}/../../../":"$PYTHONPATH"


if [[ "$PWD" != "$DIR" ]]; then
    echo "Please run the script in the script's residing directory"
    exit 0
fi


if [[ "$tag" == "" ]]; then
    tag="default"
fi


ROOT="$PWD/${name}"
log_file="${ROOT}/$tag-log.txt"

python -u -m new_dataset.anomaly_detection.train_autoencoder \
--model_spec "$ROOT/model_spec.yaml" \
--model_file "${ROOT}/$tag-model" \
--image_dir "../../../../data/new_dataset_2019/normalized" \
--max_epochs 1000 \
--vis_tag "_$tag" \
${@:3} | tee "${log_file}"
