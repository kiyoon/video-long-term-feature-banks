#!/bin/bash

python tools/test_net.py --config_file configs/epic_noun_r50_lfb_max.yaml LFB.LOAD_LFB True LFB.LOAD_LFB_PATH data/epic/noun_lfb TEST.PARAMS_FILE data/epic/noun_lfb/model_final.pkl

