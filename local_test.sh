export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PORT=29532

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}

config=projects/configs/$1.py
checkpoint=$2

if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    extra_cfg_options=()
else
    extra_cfg_options=(--cfg-options fp16=None)
fi

echo "number of gpus: "${gpu_num}
echo "config file: "${config}
echo "checkpoint: "${checkpoint}

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_test.sh \
        ${config} \
        ${checkpoint} \
        ${gpu_num} \
        --eval bbox \
        "${extra_cfg_options[@]}" \
        $@
else
    python ./tools/test.py \
        ${config} \
        ${checkpoint} \
        --eval bbox \
        "${extra_cfg_options[@]}" \
        $@
fi
