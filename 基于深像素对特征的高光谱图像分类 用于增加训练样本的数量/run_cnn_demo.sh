#! /bin/bash
mkdir checkpoint
mkdir hyper_log
python train.py
python eval_from_ckpt.py
