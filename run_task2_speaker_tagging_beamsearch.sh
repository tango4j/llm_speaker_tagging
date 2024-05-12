
branch_name="llm_diar_cbsd_main"
BASEPATH=/home/taejinp/projects/$branch_name/NeMo

export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH

### Speaker Tagging Task-2 Parameters
DIAR_LM_PATH="/disk_c/models/lm/4gram_small.arpa" # 2M entries
ASRDIAR_FILE_NAME=err_dev.short2
WORKSPACE=/disk_a/datasets/gensec_dataset/llm_diar_gensec_originalnames
INPUT_ERROR_SRC_LIST_PATH=$WORKSPACE/$ASRDIAR_FILE_NAME.src.list
GROUNDTRUTH_REF_LIST_PATH=$WORKSPACE/$ASRDIAR_FILE_NAME.ref.list
DIAR_OUT_DOWNLOAD=$WORKSPACE/short2_all_seglst_infer
# INPUT_ERROR_SRC_LIST_PATH=/disk_a/datasets/gensec_dataset/llm_diar_gensec_originalnames/err_dev.short12.list
# DIAR_OUT_DOWNLOAD=/disk_a/datasets/gensec_dataset/llm_diar_gensec_originalnames/short12_all_seglst_infer
mkdir -p $DIAR_OUT_DOWNLOAD


### SLT 2024 Speaker Tagging Setting v1.0.1
ALPHA=0.4
BETA=0.04
PARALLEL_CHUNK_WORD_LEN=250
BEAM_WIDTH=16
WORD_WINDOW=32
PEAK_PROB=0.95

### ASRDIAR_FILE_NAME=err_dev.short2
# Evaluating the original source transcript.
# Source     cpWER:  0.22208320300281514
# Evaluating the original hypothesis transcript.
# Hypothesis cpWER:  0.22051923678448546

### SLT 2024 Speaker Tagging Setting v1.0.2
ALPHA=0.4
BETA=0.04
PARALLEL_CHUNK_WORD_LEN=100
BEAM_WIDTH=16
WORD_WINDOW=32
PEAK_PROB=0.95

USE_NGRAM=True
LM_METHOD=llm


# Get the base name of the test_manifest and remove extension
UNIQ_MEMO=$(basename "${INPUT_ERROR_SRC_LIST_PATH}" .json | sed 's/\./_/g') 
echo "UNIQ MEMO:" $UNIQ_MEMO
TRIAL=telephonic
BATCH_SIZE=11


rm $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst.json
rm $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json
rm $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst.json


python $BASEPATH/examples/speaker_tasks/diarization/neural_diarizer/speaker_tagging_beamsearch.py \
    use_mp=true \
    arpa_language_model=$DIAR_LM_PATH \
    batch_size=$BATCH_SIZE \
    word_window=16 \
    groundtruth_ref_list_path=$GROUNDTRUTH_REF_LIST_PATH \
    input_error_src_list_path=$INPUT_ERROR_SRC_LIST_PATH \
    port=[5501,5502,5511,5512,5521,5522,5531,5532] \
    parallel_chunk_word_len=$PARALLEL_CHUNK_WORD_LEN \
    use_ngram=$USE_NGRAM \
    alpha=$ALPHA \
    beta=$BETA \
    beam_width=$BEAM_WIDTH \
    word_window=$WORD_WINDOW \
    peak_prob=$PEAK_PROB \
    out_dir="$DIAR_OUT_DOWNLOAD"  # Make sure this variable is correctly defined 


echo "Evaluating the original source transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Source     cpWER: " $(jq '.error_rate'  $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst_cpwer.json)

echo "Evaluating the original hypothesis transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Hypothesis cpWER: " $(jq '.error_rate'  $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst_cpwer.json)
