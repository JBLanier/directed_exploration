#!/usr/bin/env bash

python directed_exploration/a2c/run_exploration.py \
--working-dir None \
--validation-data-dir None \
--demo-debug False \
--num-env 48 \
--env-id 'BreakoutDeterministic-v4' \
--curiosity-train-sequence-length 5 \
--create-heatmaps False \
--frame-size 84 84 \
--a2c-entropy-coefficient 0.4 \
--extrinsic-reward-coefficient 1 \
--intrinsic-reward-coefficient 0 \
--num-timesteps 10e6
