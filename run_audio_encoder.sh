PYTHON=${PYTHON:-"python"}
NNODES=$((${ARNOLD_SERVER_NUM:-0} + ${ARNOLD_WORKER_NUM:-1}))
RANK=${ARNOLD_ID:-0}
MASTER_ADDR=${METIS_SERVER_0_HOST:-${METIS_WORKER_0_HOST:-'127.0.0.1'}}
MASTER_PORT=${METIS_SERVER_0_PORT:-${METIS_WORKER_0_PORT:-'29500'}}


echo $PYTHON, $NNODES, $RANK, $MASTER_ADDR, $MASTER_PORT
$PYTHON -m torch.distributed.launch --nproc_per_node=$1 \
    --nnodes=${NNODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}  \
    ./Audio_to_Image/train_audio_encoder.py ${@:2} \
    2>&1 | tee ./output/Audio_to_Image/log.txt