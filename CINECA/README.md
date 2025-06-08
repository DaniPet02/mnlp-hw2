# Cineca HPC System

### Basic Workflow

Login on 

* [Cineca - Userdb](https://userdb.hpc.cineca.it/)

Request HPC access



* Create a job script with Slurm `directives`.

* Submit the job using `sbatch`.

* Monitor the job using commands like `squeue` and `scontrol`.

* Cancel a job if needed with `scancel`.




### Example of Script shell (SLURM)

```sh

#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=32         # 32 tasks per node
#SBATCH --time=1:00:00               # time limit: 1 hour
#SBATCH --error=myJob.err            # standard error file
#SBATCH --output=myJob.out           # standard output file
#SBATCH --account=<Project Account>  # project account
#SBATCH --partition=<partition_name> # partition name
#SBATCH --qos=<qos_name>             # quality of service


./my_application
```

### Use Cineca resource outside Terminal (Visual Studio Code etc)

You can run jupyter kernel using Cineca Resources login on follow page

* https://jupyter.g100.cineca.it/hub/login?next=%2Fhub%2F




### Additional Resources

* [Official Documentation](https://docs.hpc.cineca.it/index.html)
    - [System Access](https://docs.hpc.cineca.it/general/access.html#)
    - [Job Scheduling](https://docs.hpc.cineca.it/hpc/hpc_scheduler.html#)

* [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
