#!/bin/bash
#SBATCH --job-name=WY_human
#SBATCH --nodes=2
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-22:59:00
#SBATCH --partition=normal 
#SBATCH --gres=gpu:0 
#SBATCH --output=server_logs/scripts_human_upper_run_training 
#SBATCH --error=server_logs/scripts_human_upper_run_training_error

SeedStart=$1
SeedEnd=$2
q_bound_type=$3
subsampling=$4
state_only=$5
num_demonstrations=$6
path_by_q_type=$7

#state_only=True
#num_demonstrations=5

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

declare -a q_bound_types=($q_bound_type)
#'w_bounded' )

qtypes_len=${#q_bound_types[@]}

demo_dir='demo/'

original_trainer='True'
current_date_time="`date +%Y%m%d%H%M%S`"
ep_steps=1000


store_path1="tmp/state_only_${state_only}/pwil-${path_by_q_type}/constrained_by_wasserstein_"
store_path2="/subsampling_${subsampling}_numdemo_${num_demonstrations}/${env}_random_"
current_time="${current_date_time}"
echo ${store_path}

for i in `seq $SeedStart $SeedEnd`
do	
	echo ${i} ${q_bound_type}
	srun --ntasks=1 \
	    --exclusive \
	    python -m trainer \
		--workdir="${store_path1}${q_bound_type}${store_path2}${i}/${current_time}" \
		--env_name=${env} \
		--demo_dir=$demo_dir \
		--original_trainer_type=${original_trainer} \
		--state_only=$state_only \
		--ep_steps=$ep_steps \
		--subsampling=$subsampling \
		--num_demonstrations=$num_demonstrations \
		--q_bound_type=${q_bound_type} \
		--random_seed=${i} &
	
		
done
wait

