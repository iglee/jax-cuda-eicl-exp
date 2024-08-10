#!/bin/bash
mkdir /tmp/images_all_exemplars
mkdir /tmp/images_augmented
mkdir /tmp/images_identical

python -m emergent_in_context_learning.experiment.experiment \
       --config emergent_in_context_learning/experiment/configs/images_all_exemplars.py \
       --jaxline_mode train \
       --logtostderr
mv /tmp/!(images*) /tmp/images_all_exemplars

python -m emergent_in_context_learning.experiment.experiment \
       --config emergent_in_context_learning/experiment/configs/images_augmented.py \
       --jaxline_mode train \
       --logtostderr
mv /tmp/!(images*) /tmp/images_augmented

python -m emergent_in_context_learning.experiment.experiment \
       --config emergent_in_context_learning/experiment/configs/images_identical.py \
       --jaxline_mode train \
       --logtostderr
mv /tmp/!(images*) /tmp/images_identical