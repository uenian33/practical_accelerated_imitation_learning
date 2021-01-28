#!/bin/bash


#env='HalfCheetah-v2'
#env='Humanoid-v2'
#env='Hopper-v2'
#env='Ant-v2'
#env='Walker2d-v2'
env='BipedalWalker-v3'
#env='MountainCarContinuous-v0'
#env='LunarLanderContinuous-v2'
#env='HalfCheetahBulletEnv-v0'
#env='HumanoidBulletEnv-v0'
#env='HopperBulletEnv-v0'
#env='AntBulletEnv-v0'
#env='Walker2dBulletEnv-v0'
#env='BipedalWalker-v3'

# The q constrain type specification
#['None', ,'DDPGfD','standard_lower_bound','expert_lower_bound','expert_upper_bound','target_bound','hybrid']

q_bound_type='DDPGfD'
DEMO_DIR='demo/'

original_trainer='True'
current_date_time="`date +%Y%m%d%H%M%S`"
store_path="tmp/pwil/constrained_by_${q_bound_type}/${env}_subsampling_${subsampling}_numdemo_${num_demonstrations}_${current_date_time}"
ep_steps=1000
echo ${store_path}

state_only=True
subsampling=20
num_demonstrations=5

python -m trainer \
--workdir=${store_path} \
--env_name=${env} \
 --demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type}

