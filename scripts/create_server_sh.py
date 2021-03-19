env_names = ["ant", "bipedal", "halfcheetah", "hopper", "humanoid", "walker"]
ctypes = ["upper", "nstep_lower", "nstep_lower_upper", "none", "DDPGfD", "constrained_lower", "constrained_lower_upper"]
out_sh = "submit_jobs.sh"

f= open(out_sh,"a+")
for env_name in env_names:
	for ctype in ctypes:
		sh_str = "sbatch -J WY_JOB --ntasks=1 --cpus-per-task=1 --mem=40960 --time=2-22:59:00 --partition=normal --gres=gpu:0  \
		 --output=server_logs/scripts_{}_{}_run_training \
		 --error=server_logs/scripts_{}_{}_run_training_error\
		  scripts/{}/run_training_{}.sh\n".format(env_name,ctype,env_name,ctype,env_name,ctype)

		f.write(sh_str)
		print(sh_str)
f.close()