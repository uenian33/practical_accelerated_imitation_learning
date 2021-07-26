#!/bin/bash
#SBATCH --job-name=WY_walker
#SBATCH --nodes=7
#SBATCH --ntasks=70
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-22:59:00
#SBATCH --partition=normal 
#SBATCH --gres=gpu:0 
#SBATCH --output=server_logs/scripts_walker_upper_run_training 
#SBATCH --error=server_logs/scripts_walker_upper_run_training_error

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
#env='HumanoidBulletEnv-v0'
#env='HopperBulletEnv-v0'
#env='AntBulletEnv-v0'
env='Walker2DBulletEnv-v0'
#env='BipedalWalker-v3'

declare -a q_bound_types=('upper' 'none' 'DDPGfD' 'nstep_lower' 'nstep_lower_upper' 'constrained_lower' 'constrained_lower_upper' )
qtypes_len=${#q_bound_types[@]}

demo_dir='demo/'

original_trainer='True'
current_date_time="`date +%Y%m%d%H%M%S`"
ep_steps=1000

state_only=True
subsampling=20
num_demonstrations=5

random_seed=0
store_path1="tmp/pwil/constrained_by_$"
store_path2="/${env}_subsampling_${subsampling}_numdemo_${num_demonstrations}_random_"
current_time="${current_date_time}"
echo ${store_path}

for i in `seq 1 10`
do	
	for q_bound_type in "${q_bound_types[@]}"
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
		
done
wait

