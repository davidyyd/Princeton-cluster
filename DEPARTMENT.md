# Department-wide Cluster Manual

## Configure SSH

We recommend using the remote explorer in Cursor / VSCode to establish SSH connection with the cluster. For Cursor, you can claim a free membership [here](https://www.cursor.com/pricing). Open the SSH configuration under the remote explorer.

<p align="center">
<img src="images/remote_explorer.png" width=50%
class="center">
</p>
<p align="center">
<img src="images/ssh_config.png" width=50%
class="center">
</p>
Copy and paste the following into the SSH configuration. Replace `$NetID` with the username for the account (everything before @). For example, if the account is yy8435@princeton.edu, then the NetID should be `yy8435`.  

```
Host neuronic
    HostName neuronic.cs.princeton.edu
    User $NetID
    ControlMaster auto
    ControlPersist yes
    ControlPath ~/.ssh/sockets/%p-%h-%r
```


## Connect to the Cluster

Open the Command Palette in VSCode with `Ctrl+Shift+P` (or `Command+Shift+P` in Mac). Type `>Remote-SSH:Connect to Host` and press Enter. Then type `neuronic` in the Command Palette and press Enter.   

<p align="center">
<img src="images/command_palette.png" width=50%
class="center">
</p>
<p align="center">
<img src="images/neuronic_host.png" width=50%
class="center">
</p>

It will ask you to type in the password for the account. 
<p align="center">
<img src="images/enter_password.png" width=50%
class="center">
</p>

Then complete the two-factor authentication step.
<p align="center">
<img src="images/two_factor.png" width=50%
class="center">
</p>

Check the output of SSH connection to select which two-factor login method. Here 1 is for Duo Push, 2 is for Phone Call, and 3 is for SMS Passcode.
<p align="center">
<img src="images/two_factor_output.png" width=50%
class="center">
</p>

Finally click the Open Folder button.

<p align="center">
<img src="images/open_folder.png" width=50%
class="center">
</p>
It will automatically set the path to the home directory, which should be /u/$NetID.
<p align="center">
<img src="images/neuronic_path.png" width=50%
class="center">
</p>

## Storage Space

There are three types of storage space on the cluster:

- Home space (`/u/$NetID` or `~`): 
  This is the home directory for each user. Since there is only a limit of 16GB, it should be used to store code only.
- Project space (`/n/fs/vision-mix/$NetID`): 
  This is the shared project directory for each user in our lab. It has a total limit of 11TB across all users. You can use it to store your conda environment, model checkpoints, and other large files. Please be considerate with your usage, as this space is shared by everyone. Each user’s directory needs to be created manually. To request a new one, please contact David Yin by [email](yida.yin@princeton.edu) or [messenger](https://www.facebook.com/yida.yin.5?mibextid=wwXIfr&mibextid=wwXIfr) for help.
- Scratch space (`/scratch/$NetID`): 
  This is the shared scratch directory on each node (i.e., not accessible from other nodes). It has a limit of 3.5TB across all users. You can use it to store any temporary files, such as pip install cache and huggingface cache. Note that this space is not backed up and rountinely purged, so you should not store any important files here.


## Submit your First Slurm Job

Below we show a simple example of submitting a slurm job to train a neural network on the MNIST dataset using PyTorch Distributed Data Parallel (DDP).
We first clone this repo and create a conda environment called `torch-env` to install relevant packages.

```bash
git clone https://github.com/davidyyd/Princeton-cluster.git
module purge
module load anaconda3/2024.02
source ~/.bashrc
conda create -n torch-env python=3.12
conda activate torch-env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


Download the MNIST dataset. 

```bash
python download_data.py
```

Finally use `sbatch` command to submit the job:
```bash
sbatch job_department.slurm
```

You can check the log file ``slurm-xxxxx.out`` under the root where you run the command. The model should achieve 98% accuracy on the test set in two epochs.

For more details on the above slurm script, check out [here](https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp)


## Request Interactive Session

Interactive sessions are useful for testing scripts. To start an interactive session on a GPU node, use:

```bash
salloc --nodes=1 --ntasks=1 --time=60:00 --cpus-per-task=8 --mem=32G --gres=gpu:4
```

Once the session is granted, you will be logged into a compute node with GPU access. You can now run a Python script directly without `sbatch` command:
```bash
module purge
module load anaconda3/2024.02
conda activate torch-env
python -m torch.distributed.run --nproc_per_node=4 mnist_classify_ddp.py --epochs 2
``` 

To exit the interactive session, simply type:

```bash
exit
```

Note you can also use the login node to do small-scale testing. Since this is a shared space for all users, any work consuming too many CPU / GPU resources will be killed automatically.

## Common Commands

Here we list some common commands for managing your job submissions and account usage. If you have used SLURM before, you can skip this section.

### Check status of jobs

`squeue` is used to check the status of all your submitted jobs. You can check the status of your own jobs by:
```bash
squeue -u $NetID
```

It will print out the status of all your submitted jobs, including the job ID, partition, name, user, status, time, number of nodes, and node list:
```bash
  JOBID PARTITION           NAME     USER ST       TIME  NODES NODELIST(REASON)
2251339       all classification   yy8435  R 1-04:11:33      4 (Priority)
2251237       all           gpt2   yy8435  R 1-04:11:33      2 neu[329-330]
```

It is also possible to check the status of all jobs in the cluster by removing the `-u $NetID` option:
```bash
squeue
```

### Cancel a job
`scancel` is used to cancel job in the cluster. You can cancel a job by its job ID:
```bash
scancel $job_id
```
It is also possible to cancel all your jobs at once by specifying the NetID:
```bash
scancel -u $NetID
```

### Check current GPU status in the cluster

`gpudash` is a tool to check the GPU status across all nodes. 
```bash
gpudash
```
It will print out the gpu utilization for each gpu in the cluster during the last hour:
```bash
                                    NEURONIC-GPU UTILIZATION (Mon Jul 28)

            3:10 AM       3:20 AM       3:30 AM       3:40 AM       3:50 AM       4:00 AM       4:10 AM
neu301 0   yy8435:100    yy8435:99     yy8435:100    yy8435:100    yy8435:98     yy8435:100    yy8435:100
       1   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:99     yy8435:100
       2   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:98     yy8435:100    yy8435:100
       3   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100
       4   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100
       5   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:99     yy8435:100    yy8435:100
       6   yy8435:99     yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100
       7   yy8435:99     yy8435:100    yy8435:100    yy8435:100    yy8435:99     yy8435:100    yy8435:100
...
neu332 0   yy8435:100    yy8435:99     yy8435:100    yy8435:100    yy8435:98     yy8435:100    yy8435:100
       1   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:99     yy8435:100
       2   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:98     yy8435:100    yy8435:100
       3   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100
       4   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100
       5   yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:99     yy8435:100    yy8435:100
       6   yy8435:99     yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100    yy8435:100
       7   yy8435:99     yy8435:100    yy8435:100    yy8435:100    yy8435:99     yy8435:100    yy8435:100
            3:10 AM       3:20 AM       3:30 AM       3:40 AM       3:50 AM       4:00 AM       4:10 AM
```

### Check CPU and GPU hours

`sreport` is used to check the CPU and GPU hours of your account. You need to specify the start and end date (in the format YYYY-MM-DD) for the report.
```bash
sreport -t Hours -T CPU,gres/gpu cluster AccountUtilizationByUser Users=$NetID Start=$start_date End=$end_date
```
It will print out the report as follows:
```bash
Usage reported in TRES Hours
--------------------------------------------------------------------------------
  Cluster         Account     Login     Proper Name      TRES Name     Used
--------- --------------- --------- --------------- -------------- --------
 neuronic            seas    yy8435        Yida Yin            cpu     8722
 neuronic            seas    yy8435        Yida Yin       gres/gpu     1034
```

### Check priority of your account

`sshare` is used to check the priority of your account.
```bash
sshare -u $NetID
```
It will print out something like this:
```bash
Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare
-------------------- ---------- ---------- ----------- ----------- ------------- ----------
root                                          0.000000 6744067132527      1.000000
 seas                                    1    0.500000 6744067132527      1.000000
  seas                   yy8435          1    0.004762 37444021933      0.005552   0.109005
```

Only the last row is relevant. `NormShares` reflects your account’s allocated share of priority, equally divided among all users. `RawUsage` and `EffectiveUsage` represent your account's actual resource usage, decayed over time with a 14-day half-life. These values are less informative than those provided by `sreport`. `FairShare` indicates your current scheduling priority. It starts at 1.0 for new accounts and decreases as you consume more resources over time. It directly impacts how your jobs are prioritized in the queue.

### Other helpful commands (still under development)

We also create additional helpful commands for managing your jobs and files under ```/n/fs/vision-mix/helpful_commands```.


To see the current status of all the nodes in the cluster, you can use the following command:

```bash
bash check_all_nodes.sh
```

It will print out the number of free CPUs, CPU memory usage, and the number of free GPUs for each of 32 nodes in neuronic. You can use this command to determine best resources for your jobs.

```bash
neu301     FreeCPUs= 40/104   FreeMem=375.0GiB/503.0GiB   FreeGPUs=1/8
...
neu332     FreeCPUs=  4/104   FreeCPUMem= 23.0GiB/503.0GiB   FreeGPUs=4/8
```