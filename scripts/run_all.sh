

SeedStart=$1
SeedEnd=$2
q_bound_type=$3
subsampling=$4
state_only=$5
num_demonstrations=$6
path_by_q_type=$7


sbatch scripts/simple_bound_scripts/Walker.sh   ${SeedStart}  ${SeedEnd}  ${q_bound_type} ${subsampling} ${state_only} ${num_demonstrations} ${path_by_q_type}  &

sbatch scripts/simple_bound_scripts/Ant.sh   ${SeedStart}  ${SeedEnd}  ${q_bound_type} ${subsampling} ${state_only} ${num_demonstrations} ${path_by_q_type}  &

sbatch scripts/simple_bound_scripts/Bipedal.sh   ${SeedStart}  ${SeedEnd}  ${q_bound_type} ${subsampling} ${state_only} ${num_demonstrations} ${path_by_q_type}  &

sbatch scripts/simple_bound_scripts/Humanoid.sh   ${SeedStart}  ${SeedEnd}  ${q_bound_type} ${subsampling} ${state_only} ${num_demonstrations} ${path_by_q_type}  &

sbatch scripts/simple_bound_scripts/Hopper.sh   ${SeedStart}  ${SeedEnd}  ${q_bound_type} ${subsampling} ${state_only} ${num_demonstrations} ${path_by_q_type}  &

sbatch scripts/simple_bound_scripts/Halfcheetah.sh   ${SeedStart}  ${SeedEnd}  ${q_bound_type} ${subsampling} ${state_only} ${num_demonstrations} ${path_by_q_type}  


#sbatch scripts/simple_bound_scripts/Walker.sh  1  10  none 20 True 5 test_none &
