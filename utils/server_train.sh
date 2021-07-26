#!/bin/bash
#!/bin/bash
#module load matlab
module load CUDA/10.1
source activate pwil 
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
# !!!!SBATCH -J WY_JOB
# !!!!SBATCH --output=log_fcn_se_siamese_%a.txt
# !!!!SBATCH --error=log_fcn_se_siamese_%a.txt
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

#module load matlab
module load CUDA/10.1
source activate pwil 

# Finally run your job. Here's an example of a python script.
#!/bin/bash


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
env='HopperBulletEnv-v0'
#env='AntBulletEnv-v0'
#env='Walker2dBulletEnv-v0'
#env='BipedalWalker-v3'

# The q constrain type specification
#['None', ,'DDPGfD','standard_lower_bound','expert_lower_bound','expert_upper_bound','target_bound','hybrid']

q_bound_type='DDPGfD'
DEMO_DIR='demo/'

original_trainer='True'
current_date_time="`date +%Y%m%d%H%M%S`"
store_path="tmp/pwil2/constrained_by_${q_bound_type}/${env}_subsampling_${subsampling}_numdemo_${num_demonstrations}/${current_date_time}"
ep_steps=500
echo ${store_path}

state_only=True
subsampling=20
num_demonstrations=5

random_seed=0

python -m trainer \
--workdir=${store_path} \
--env_name=${env} \
--demo_dir=$DEMO_DIR \
--original_trainer_type=${original_trainer} \
--state_only=$state_only \
--ep_steps=$ep_steps \
--subsampling=$subsampling \
--num_demonstrations=$num_demonstrations \
--q_bound_type=${q_bound_type} \
--random_seed=$random_seed

# sbatch -J WY_JOB --ntasks=1 --cpus-per-task=4 --mem=71680 --time=1-22:59:00 --partition=gpu --gres=gpu:2 --output=log_triplet_%a.txt --error=log_triplet_%a.txt train.sh