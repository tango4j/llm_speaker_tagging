import hydra
from typing import List, Optional
from dataclasses import dataclass, field
import kenlm
from beam_search_utils import (
    SpeakerTaggingBeamSearchDecoder,
    load_input_jsons,
    load_reference_jsons,
    write_seglst_jsons,
    run_mp_beam_search_decoding,
    convert_nemo_json_to_seglst,
)
from hydra.core.config_store import ConfigStore

__INFO_TAG__ = "[INFO]"

@dataclass
class RealigningLanguageModelParameters:
    batch_size: int = 32
    use_mp: bool = True
    input_error_src_list_path: Optional[str] = None
    groundtruth_ref_list_path: Optional[str] = None
    arpa_language_model: Optional[str] = None
    word_window: int = 32
    port: List[int] = field(default_factory=list)
    parallel_chunk_word_len: int = 250
    use_ngram: bool = True
    peak_prob: float = 0.95
    alpha: float = 0.5
    beta: float = 0.05
    beam_width: int = 16
    out_dir: Optional[str] = None

cs = ConfigStore.instance()
cs.store(name="config", node=RealigningLanguageModelParameters)

@hydra.main(config_name="config", version_base="1.1")
def main(cfg: RealigningLanguageModelParameters) -> None:
    trans_info_dict = load_input_jsons(input_error_src_list_path=cfg.input_error_src_list_path, peak_prob=float(cfg.peak_prob))
    reference_info_dict  = load_reference_jsons(reference_seglst_list_path=cfg.groundtruth_ref_list_path)
    source_info_dict = load_reference_jsons(reference_seglst_list_path=cfg.input_error_src_list_path)
    loaded_kenlm_model = kenlm.Model(cfg.arpa_language_model)
    
    speaker_beam_search_decoder = SpeakerTaggingBeamSearchDecoder(loaded_kenlm_model=loaded_kenlm_model, cfg=cfg)
    
    div_trans_info_dict = speaker_beam_search_decoder.divide_chunks(trans_info_dict=trans_info_dict, 
                                                                    win_len=cfg.parallel_chunk_word_len, 
                                                                    word_window=cfg.word_window,
                                                                    port=cfg.port,)
    
    trans_info_dict = run_mp_beam_search_decoding(speaker_beam_search_decoder, 
                                                    loaded_kenlm_model=loaded_kenlm_model,
                                                    trans_info_dict=div_trans_info_dict, 
                                                    org_trans_info_dict=trans_info_dict, 
                                                    div_mp=True,
                                                    win_len=cfg.parallel_chunk_word_len,
                                                    word_window=cfg.word_window,
                                                    port=cfg.port,
                                                    use_ngram=cfg.use_ngram,
                                                    )
    hypothesis_sessions_dict = convert_nemo_json_to_seglst(trans_info_dict) 
    
    write_seglst_jsons(hypothesis_sessions_dict, input_error_src_list_path=cfg.input_error_src_list_path, diar_out_path=cfg.out_dir, ext_str='hyp')
    write_seglst_jsons(reference_info_dict, input_error_src_list_path=cfg.groundtruth_ref_list_path, diar_out_path=cfg.out_dir, ext_str='ref')
    write_seglst_jsons(source_info_dict, input_error_src_list_path=cfg.groundtruth_ref_list_path, diar_out_path=cfg.out_dir, ext_str='src')
    print(f"{__INFO_TAG__} Parameters used: \
            \n ALPHA: {cfg.alpha} \
            \n BETA: {cfg.beta} \
            \n BEAM WIDTH: {cfg.beam_width} \
            \n Word Window: {cfg.word_window} \
            \n Use Ngram: {cfg.use_ngram} \
            \n Chunk Word Len: {cfg.parallel_chunk_word_len} \
            \n SpeakerLM Model: {cfg.arpa_language_model}") \

if __name__ == '__main__':
    main()