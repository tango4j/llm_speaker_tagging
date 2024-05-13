
### Speaker Tagging Task-2 Parameters


BASEPATH=${PWD}
DIAR_LM_PATH=$BASEPATH/arpa_model/4gram_small.arpa
ASRDIAR_FILE_NAME=err_dev
OPTUNA_STUDY_NAME=speaker_beam_search_${ASRDIAR_FILE_NAME}
WORKSPACE=$BASEPATH/SLT-Task2-Post-ASR-Speaker-Tagging
INPUT_ERROR_SRC_LIST_PATH=$BASEPATH/$ASRDIAR_FILE_NAME.src.list
GROUNDTRUTH_REF_LIST_PATH=$BASEPATH/$ASRDIAR_FILE_NAME.ref.list
DIAR_OUT_DOWNLOAD=$WORKSPACE/$ASRDIAR_FILE_NAME
mkdir -p $DIAR_OUT_DOWNLOAD


### SLT 2024 Speaker Tagging Setting v1.0.2
ALPHA=0.4
BETA=0.04
PARALLEL_CHUNK_WORD_LEN=100
BEAM_WIDTH=16
WORD_WINDOW=32
PEAK_PROB=0.95
USE_NGRAM=True
LM_METHOD=ngram

# Get the base name of the test_manifest and remove extension
UNIQ_MEMO=$(basename "${INPUT_ERROR_SRC_LIST_PATH}" .json | sed 's/\./_/g') 
echo "UNIQ MEMO:" $UNIQ_MEMO
TRIAL=telephonic
BATCH_SIZE=11


rm $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst.json
rm $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json
rm $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst.json


python $BASEPATH/speaker_tagging_beamsearch.py \
    hyper_params_optim=false \
    port=[5501,5502,5511,5512,5521,5522,5531,5532] \
    arpa_language_model=$DIAR_LM_PATH \
    batch_size=$BATCH_SIZE \
    groundtruth_ref_list_path=$GROUNDTRUTH_REF_LIST_PATH \
    input_error_src_list_path=$INPUT_ERROR_SRC_LIST_PATH \
    parallel_chunk_word_len=$PARALLEL_CHUNK_WORD_LEN \
    use_ngram=$USE_NGRAM \
    alpha=$ALPHA \
    beta=$BETA \
    beam_width=$BEAM_WIDTH \
    word_window=$WORD_WINDOW \
    peak_prob=$PEAK_PROB \



echo "Evaluating the original source transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Source     cpWER: " $(jq '.error_rate' "[ $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst_cpwer.json) ]"

echo "Evaluating the original hypothesis transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Hypothesis cpWER: " $(jq '.error_rate'  $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst_cpwer.json)
