import fnmatch
import os

matches = []

slurm_base = "sbatch -J WY_JOB --ntasks=1 --cpus-per-task=1 --mem=4096 --time=2-22:59:00 --partition=normal --gres=gpu:0 "
for root, dirnames, filenames in os.walk('scripts'):
    for filename in fnmatch.filter(filenames, '*'):
        matches.append(os.path.join(root, filename))
        output_des = " --output=server_logs/" + os.path.join(root, filename).replace("/", "_").replace(".sh", "")
        err_output_des = " --error=server_logs/" + os.path.join(root, filename).replace("/", "_").replace(".sh", "") + "_error"
        script_des = " " + os.path.join(root, filename)

        executr_command = slurm_base + output_des + err_output_des + script_des

        try:
            """
            # with is like your try .. finally block in this case
            with open(os.path.join(root, filename), 'r') as file:
                # read a list of lines into data
                data = file.readlines()

            # Read in the file
            with open(os.path.join(root, filename), 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace('tmp/pwil2/', 'tmp/pwil2/')

            # Write the file out again
            with open(os.path.join(root, filename), 'w') as file:
                file.write(filedata)
            """

            """
            # now change the 2nd line, note that you have to add a newline
            new_line = "#!/bin/bash\n#module load matlab\nmodule load CUDA/10.1\nsource activate pwil \n"
            
            data[1] = new_line

            # and write everything back
            with open(os.path.join(root, filename), 'w') as file:
                file.writelines( data )
            """
            if "random0" in script_des:
                with open("submit_jobs", 'a') as file:
                    file.writelines(executr_command + "\n")
            else:
                with open("submit_jobs2", 'a') as file:
                    file.writelines(executr_command + "\n")
                    file.writelines("\n")

            print(executr_command + "\n")

        except:
            True
