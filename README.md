# llm_speaker_tagging

SLT 2024 Challenge: Post-ASR-Speaker-Tagging Baseline



# Project Name

SLT 2024 Challenge GenSEC Track 2: Post-ASR-Speaker-Tagging Baseline 

## Featuresimag

- Data download and cleaning
- n-gram + beam search decoder based baselinee system 

## Installation

Run the following commands at the main level of this repository.

### Conda Environment

```
conda create --name llmspk python=3.10
```

## Baseline System: Contextudal Beam Search Decoding

![BSD Equation](images/bsd_equation.png)


![Overall Dataflow](images/overall_dataflow.png)

![Two Realms](images/two_realms.png)

![Word Level Speaker Probability](images/word_level_spk_prob.png)


### Install requirements

You need to install the following packages 

```
kenlm
arpa
numpy
hydra-core
meeteval
tqdm
requests
simplejson
pydiardecode @ git+https://github.com/tango4j/pydiardecode@main
```

Simply install all the requirments. 

```
pip install -r requirements.txt
```

### Download ARPA language model

```
mkdir -p arpa_model
cd arpa_model
wget https://kaldi-asr.org/models/5/4gram_small.arpa.gz
gunzip 4gram_small.arpa.gz
```

### Download track-2 challenge dev set and eval set 

Clone the dataset from Hugging Face server.
```
git clone https://huggingface.co/datasets/GenSEC-LLM/SLT-Task2-Post-ASR-Speaker-Tagging
```

```
find $PWD/SLT-Task2-Post-ASR-Speaker-Tagging/err_source_text/dev -maxdepth 1 -type f -name "*.seglst.json" > err_dev.src.list
find $PWD/SLT-Task2-Post-ASR-Speaker-Tagging/ref_annotated_text/dev -maxdepth 1 -type f -name "*.seglst.json" > err_dev.ref.list
```

### Launch the baseline script

Now you are ready to launch the script.
Launch the baseline script `run_speaker_tagging_beam_search.sh`

```
BASEPATH=${PWD}
DIAR_LM_PATH=$BASEPATH/arpa_model/4gram_small.arpa
ASRDIAR_FILE_NAME=err_dev
WORKSPACE=$BASEPATH/SLT-Task2-Post-ASR-Speaker-Tagging
INPUT_ERROR_SRC_LIST_PATH=$BASEPATH/$ASRDIAR_FILE_NAME.src.list
GROUNDTRUTH_REF_LIST_PATH=$BASEPATH/$ASRDIAR_FILE_NAME.ref.list
DIAR_OUT_DOWNLOAD=$WORKSPACE/short2_all_seglst_infer
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
    out_dir=$DIAR_OUT_DOWNLOAD 
```

### Evaluate 

We use [MeetEval](https://github.com/fgnt/meeteval) software to evaluate `cpWER`.
cpWER measures both speaker tagging and word error rate (WER) by testing all the permutation of trancripts and choosing the permutation that 
gives the lowest error.

```
echo "Evaluating the original source transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Source     cpWER: " $(jq '.error_rate' "[ $WORKSPACE/$ASRDIAR_FILE_NAME.src.seglst_cpwer.json) ]"

echo "Evaluating the original hypothesis transcript."
meeteval-wer cpwer -h $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst.json -r $WORKSPACE/$ASRDIAR_FILE_NAME.ref.seglst.json 
echo "Hypothesis cpWER: " $(jq '.error_rate'  $WORKSPACE/$ASRDIAR_FILE_NAME.hyp.seglst_cpwer.json)
```

### Reference

@inproceedings{park2024enhancing,
  title={Enhancing speaker diarization with large language models: A contextual beam search approach},
  author={Park, Tae Jin and Dhawan, Kunal and Koluguri, Nithin and Balam, Jagadeesh},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10861--10865},
  year={2024},
  organization={IEEE}
}