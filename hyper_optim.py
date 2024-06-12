import optuna
import os
import tempfile
import time
import json
import subprocess
import logging
from beam_search_utils import (
    write_seglst_jsons,
    run_mp_beam_search_decoding,
    convert_nemo_json_to_seglst,
    SpeakerTaggingBeamSearchDecoder,
)

from speaker_tagging_cpwer_jsons import process_session_data

def evaluate(cfg, temp_out_dir, asrdiar_file_name, source_info_dict, hypothesis_sessions_dict, reference_info_dict):
    write_seglst_jsons(hypothesis_sessions_dict, input_error_src_list_path=cfg.input_error_src_list_path, diar_out_path=temp_out_dir, ext_str='hyp')
    write_seglst_jsons(reference_info_dict, input_error_src_list_path=cfg.groundtruth_ref_list_path, diar_out_path=temp_out_dir, ext_str='ref')
    write_seglst_jsons(source_info_dict, input_error_src_list_path=cfg.groundtruth_ref_list_path, diar_out_path=temp_out_dir, ext_str='src')

    # Construct the file paths
    # src_seglst_json = os.path.join(temp_out_dir, f"{asrdiar_file_name}.src.seglst.json")
    hyp_seglst_json = os.path.join(temp_out_dir, f"{asrdiar_file_name}.hyp.seglst.json")
    ref_seglst_json = os.path.join(temp_out_dir, f"{asrdiar_file_name}.ref.seglst.json")
    
    # Construct the output JSON file path
    output_cpwer_hyp_json_file = os.path.join(temp_out_dir, f"{asrdiar_file_name}.hyp.seglst_cpwer.json")
    # output_cpwer_src_json_file = os.path.join(temp_out_dir, f"{asrdiar_file_name}.src.seglst_cpwer.json")

    # Run meeteval-wer command
    cmd_hyp = [
        "meeteval-wer", 
        "cpwer",
        "-h", hyp_seglst_json,
        "-r", ref_seglst_json
    ]
    subprocess.run(cmd_hyp)

    # Read the JSON file and print the cpWER
    try:
        with open(output_cpwer_hyp_json_file, "r") as file:
            data_h = json.load(file)
            print("Hypothesis cpWER:", data_h["error_rate"])
        cpwer = data_h["error_rate"]
        logging.info(f"-> HYPOTHESIS cpWER={cpwer:.4f}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Output JSON: {output_cpwer_hyp_json_file}\nfile not found.")

    return cpwer

def evaluate_diff(cfg, temp_out_dir, asrdiar_file_name, source_info_dict, hypothesis_sessions_dict, reference_info_dict):
    write_seglst_jsons(hypothesis_sessions_dict, input_error_src_list_path=cfg.input_error_src_list_path, diar_out_path=temp_out_dir, ext_str='hyp')
    write_seglst_jsons(reference_info_dict, input_error_src_list_path=cfg.groundtruth_ref_list_path, diar_out_path=temp_out_dir, ext_str='ref')
    write_seglst_jsons(source_info_dict, input_error_src_list_path=cfg.groundtruth_ref_list_path, diar_out_path=temp_out_dir, ext_str='src')

    # Construct the file paths
    src_seglst_json = os.path.join(temp_out_dir, f"{asrdiar_file_name}.src.seglst.json")
    hyp_seglst_json = os.path.join(temp_out_dir, f"{asrdiar_file_name}.hyp.seglst.json")
    ref_seglst_json = os.path.join(temp_out_dir, f"{asrdiar_file_name}.ref.seglst.json")
    
    # Run meeteval-wer command
    cmd_hyp = [
        "meeteval-wer", 
        "cpwer",
        "-h", hyp_seglst_json,
        "-r", ref_seglst_json
    ]
    subprocess.run(cmd_hyp)

    cmd_src = [
        "meeteval-wer", 
        "cpwer",
        "-h", src_seglst_json,
        "-r", ref_seglst_json
    ]
    subprocess.run(cmd_src)
    # Construct the output JSON file path
    output_cpwer_hyp_json_file = os.path.join(temp_out_dir, f"{asrdiar_file_name}.hyp.seglst_cpwer.json")
    output_cpwer_src_json_file = os.path.join(temp_out_dir, f"{asrdiar_file_name}.src.seglst_cpwer.json")
    output_cpwer_hyp_json_file_per_reco = os.path.join(temp_out_dir, f"{asrdiar_file_name}.hyp.seglst_cpwer_per_reco.json")
    output_cpwer_src_json_file_per_reco = os.path.join(temp_out_dir, f"{asrdiar_file_name}.src.seglst_cpwer_per_reco.json")

    avg_cpwer_diff = process_session_data(output_cpwer_hyp_json_file_per_reco, output_cpwer_src_json_file_per_reco)

    try:
        with open(output_cpwer_hyp_json_file, "r") as file:
            data_h = json.load(file)
        hyp_cpwer = data_h["error_rate"]
        logging.info(f"-> HYPOTHESIS cpWER={hyp_cpwer:.4f}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Output JSON: {output_cpwer_hyp_json_file}\nfile not found.")
    
    try:
        with open(output_cpwer_src_json_file, "r") as file:
            data_h = json.load(file)
        src_cpwer = data_h["error_rate"]
        logging.info(f"-> SOURCE cpWER={src_cpwer:.4f}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Output JSON: {output_cpwer_src_json_file}\nfile not found.")
    diff_cpwer = (hyp_cpwer - src_cpwer)
    logging.info(f"-> Average cpWER DIFF={avg_cpwer_diff:.4f}")
    logging.info(f"-> HYPOTHESIS Improved cpWER={diff_cpwer:.4f}")
    return diff_cpwer


def optuna_suggest_params(cfg, trial):
    cfg.alpha = trial.suggest_float("alpha", 0.5, 1.5)
    cfg.beta = trial.suggest_float("beta", 0.02, 0.4)
    cfg.beam_width = trial.suggest_int("beam_width", 2, 12)
    cfg.word_window = trial.suggest_int("word_window", 10, 50, step=10)
    cfg.use_ngram = True
    cfg.parallel_chunk_word_len = trial.suggest_int("parallel_chunk_word_len", 50, 250, step=25)
    cfg.peak_prob = trial.suggest_float("peak_prob", 0.96, 0.96)
    return cfg

def beamsearch_objective(
    trial,
    cfg,
    speaker_beam_search_decoder,
    loaded_kenlm_model,
    org_trans_info_dict,
    source_info_dict,
    reference_info_dict, 
    ):
    with tempfile.TemporaryDirectory(dir=cfg.temp_out_dir, prefix="GenSEC_") as local_temp_out_dir:
        start_time2 = time.time()
        
        if trial is not None:
            cfg = optuna_suggest_params(cfg, trial)
        speaker_beam_search_decoder = SpeakerTaggingBeamSearchDecoder(loaded_kenlm_model=loaded_kenlm_model, cfg=cfg) 
        div_trans_info_dict = speaker_beam_search_decoder.divide_chunks(trans_info_dict=org_trans_info_dict, 
                                                                        win_len=cfg.parallel_chunk_word_len, 
                                                                        word_window=cfg.word_window,
                                                                        limit_max_spks=cfg.limit_max_spks,
                                                                        port=cfg.port,)
        result_trans_info_dict = run_mp_beam_search_decoding(speaker_beam_search_decoder, 
                                                        loaded_kenlm_model=loaded_kenlm_model,
                                                        div_trans_info_dict=div_trans_info_dict, 
                                                        org_trans_info_dict=org_trans_info_dict, 
                                                        div_mp=True,
                                                        win_len=cfg.parallel_chunk_word_len,
                                                        word_window=cfg.word_window,
                                                        limit_max_spks=cfg.limit_max_spks,
                                                        port=cfg.port,
                                                        use_ngram=cfg.use_ngram,
                                                        )
        hypothesis_sessions_dict = convert_nemo_json_to_seglst(result_trans_info_dict) 
        cpwer = evaluate_diff(cfg, local_temp_out_dir, cfg.asrdiar_file_name, source_info_dict, hypothesis_sessions_dict, reference_info_dict)
    logging.info(f"Beam Search time taken for trial {trial}: {(time.time() - start_time2)/60:.2f} mins")
    if trial is not None:
        logging.info(f"Trial: {trial.number}")
    logging.info(f"[ cpWER={cpwer:.4f} ]")
    logging.info("-----------------------------------------------")
    return cpwer


def optuna_hyper_optim(
    cfg,
    speaker_beam_search_decoder, 
    loaded_kenlm_model,
    # div_trans_info_dict,
    org_trans_info_dict,
    source_info_dict,
    reference_info_dict,
    ):
    """
    Optuna hyper-parameter optimization function.

    Parameters:
        cfg (dict): A dictionary containing the configuration parameters.

    """
    worker_function = lambda trial: beamsearch_objective(    # noqa: E731
        trial=trial,
        cfg=cfg,
        speaker_beam_search_decoder=speaker_beam_search_decoder,
        loaded_kenlm_model=loaded_kenlm_model,
        # div_trans_info_dict=div_trans_info_dict,
        org_trans_info_dict=org_trans_info_dict,
        source_info_dict=source_info_dict,
        reference_info_dict=reference_info_dict, 
    )
    study = optuna.create_study(
        direction="minimize", 
        study_name=cfg.optuna_study_name, 
        storage=cfg.storage, 
        load_if_exists=True
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Setup the root logger.
    if cfg.output_log_file is not None:
        logger.addHandler(logging.FileHandler(cfg.output_log_file, mode="a"))
    logger.addHandler(logging.StreamHandler())
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    study.optimize(worker_function, n_trials=cfg.optuna_n_trials)