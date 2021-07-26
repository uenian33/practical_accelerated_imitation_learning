#!/bin/bash
#module load matlab
module load CUDA/10.1
source activate pwil 

#env='HalfCheetah-v2'
#env='Humanoid-v2'
#env='Hopper-v2'
#env='Ant-v2'
#env='Walker2d-v2'
#env='BipedalWalker-v3'
#env='MountainCarContinuous-v0'
#env='LunarLanderContinuous-v2'
#env='HalfCheetahBulletEnv-v0'
env='HumanoidBulletEnv-v0'
#env='HopperBulletEnv-v0'
#env='AntBulletEnv-v0'
#env='Walker2DBulletEnv-v0'
#env='BipedalWalker-v3'

# The q constrain type specification
#['None', ,'DDPGfD','standard_lower_bound','expert_lower_bound','expert_upper_bound','target_bound','hybrid']

q_bound_type='nstep_lower' #'constrained_lower_upper' 'nstep_lower' 'nstep_lower_upper', nstep_lower_interaction_upper, bellman_lower_upper
DEMO_DIR='demo/'

original_trainer='True'
current_date_time="`date +%Y%m%d%H%M%S`"
ep_steps=1000

state_only=True
subsampling=20
num_demonstrations=5

random_seed=0
store_path="tmp/pwil/constrained_by_${q_bound_type}/${env}_subsampling_${subsampling}_numdemo_${num_demonstrations}_random_"
current_time="${current_date_time}"
echo ${store_path}

python -m trainer \
--workdir="${store_path}0/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=0;  python -m trainer \
--workdir="${store_path}10/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=10; python -m trainer \
--workdir="${store_path}20/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=20; python -m trainer \
--workdir="${store_path}30/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=30; python -m trainer \
--workdir="${store_path}40/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=40;  python -m trainer \
--workdir="${store_path}50/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=50; python -m trainer \
--workdir="${store_path}60/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=60; python -m trainer \
--workdir="${store_path}70/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=70;  python -m trainer \
--workdir="${store_path}80/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=80; python -m trainer \
--workdir="${store_path}90/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=90;  python -m trainer \
--workdir="${store_path}100/${current_time}" \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=100; 