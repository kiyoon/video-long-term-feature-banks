#!/bin/bash

#python tools/test_net.py --config_file configs/epic_verb_r50_lfb_max.yaml LFB.MODEL_PARAMS_FILE data/epic/verb_models/epic_verb_r50_baseline.pkl TEST.PARAMS_FILE data/epic/verb_models/model_final.pkl
python tools/test_net.py --config_file configs/epic_verb_r50_lfb_max.yaml LFB.LOAD_LFB True LFB.LOAD_LFB_PATH data/epic/verb_lfb TEST.PARAMS_FILE data/epic/verb_models/model_final.pkl
