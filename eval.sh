MODEL1_CONFIG="DeepGlobalRegistration/network_configs/model1.yaml"
MODEL1_MODEL_PATH="20230910-dgr-train-adam-expLR.01-gamma0.99/checkpoint_20.pth"

MODEl2_CONFIG="DeepGlobalRegistration/network_configs/model2.yaml"
MODEL2_MODEl_PATH="20230910-dgr-train-adam-expLR.001-gamma0.99/checkpoint_20.pth"

MODEL3_CONFIG="DeepGlobalRegistration/network_configs/model3.yaml"
MODEL3_MODEl_PATH="20230910-dgr-train-adam-expLR.001-gamma0.9/checkpoint_20.pth"

NOISE="jitter"
OVERLAP="0.2"
MODEL="model3"
COMPUTE=true
EVAL=true

if [[ $MODEL == "model1" ]]; then
    NETWORK_CONFIG=$MODEL1_CONFIG
    MODEL_PATH=$MODEL1_MODEL_PATH
elif [[ $MODEL == "model2" ]]; then
    NETWORK_CONFIG=$MODEL2_CONFIG
    MODEL_PATH=$MODEL2_MODEL_PATH
elif [[ $MODEL == "model3" ]]; then
    NETWORK_CONFIG=$MODEL3_CONFIG
    MODEL_PATH=$MODEL3_MODEL_PATH
fi

CONFIG_FOLDER="DeepGlobalRegistration/mbes_data/configs/tests/meters"
MBES_CONFIG="$CONFIG_FOLDER/$NOISE/mbesdata_${NOISE}_meters_pairoverlap=$OVERLAP.yaml"
RESULTS_ROOT="20230711-$NOISE-meters-pairoverlap=$OVERLAP/${MODEL_PATH}"
mkdir -p $RESULTS_ROOT

logname="$RESULTS_ROOT/mbes_test-$NOISE-$OVERLAP-$(basename $NETWORK_CONFIG .yaml).log"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="

if [[ $COMPUTE == true ]]; then
    echo "Running mbes_test.py on noise=$NOISE overlap=$OVERLAP, network=$NETWORK_CONFIG..."
    echo "Using mbes_config=$MBES_CONFIG..."
    echo "logging to $logname..."

    python DeepGlobalRegistration/mbes_test.py \
        --mbes_config  $MBES_CONFIG\
        --network_config $NETWORK_CONFIG \
        | tee $logname
fi

if [[ $EVAL == true ]]; then
    echo "======================================="
    echo "Evaluating results at $RESULTS_ROOT..."
    python mbes-registration-data/src/evaluate_results.py \
        --results_root $RESULTS_ROOT \
        --use_transforms pred \
        | tee $RESULTS_ROOT/eval-res-$NOISE-$OVERLAP.log
fi
echo "Done!"
echo $(date "+%Y%m%d-%H-%M-%S")
echo "======================================="