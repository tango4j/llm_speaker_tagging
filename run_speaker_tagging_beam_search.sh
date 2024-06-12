
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

python $BASEPATH/speaker_tagging_beamsearch.py \
    asrdiar_file_name=$ASRDIAR_FILE_NAME \
    hyper_params_optim=false \
    arpa_language_model=$BASEPATH/arpa_model/4gram_small.arpa \
    groundtruth_ref_list_path=$BASEPATH/$ASRDIAR_FILE_NAME.ref.list \
    input_error_src_list_path=$BASEPATH/$ASRDIAR_FILE_NAME.src.list \
    alpha=0.7378102172641824 \
    beta=0.029893025590158093 \
    beam_width=9 \
    word_window=50 \
    parallel_chunk_word_len=175 \
    out_dir=$WORKSPACE \
    peak_prob=0.96 || exit 1


echo "Evaluating the original source transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Source     cpWER: " $(jq '.error_rate'  $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst_cpwer.json)

echo "Evaluating the original hypothesis transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Hypothesis cpWER: " $(jq '.error_rate'  $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst_cpwer.json)
