#!/bin/bash


#env='HalfCheetah-v2'
#env='Hopper-v2'
#env = 'HalfCheetah-v2'
env='BipedalWalker-v3'
#env='MountainCarContinuous-v0'
#env='LunarLanderContinuous-v2'
DEMO_DIR='demo/'

original_trainer='False'
store_path="tmp/pwil/${env}${path_spec}${original_trainer}"
ep_steps=1000
echo ${store_path}

state_only=False
subsampling=1

python -m trainer \
--workdir=${store_path} \
--env_name=${env} \
 --demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling