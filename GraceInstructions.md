# Texas A&M HPRC: Grace Quick Start Guide

## Grace Usage Policies
Accessing Grace requires compliance with all Texas A&M HPRC policies and Grace-specific operational guidelines.

---

## Accessing Grace

Remote login requires Duo Two-Factor Authentication (2FA) and a secure shell (SSH) session.

* **Hostname:** `grace.hprc.tamu.edu`
* **Port:** `22`
* **Password:** Same credentials as your Howdy login (input will not display characters on screen).

### Connection Methods by OS

* **Mac / Linux / Unix Terminal:**
  ```bash
  ssh [NetID]@grace.hprc.tamu.edu
  ```
* **Windows:**
  * **MobaXterm (Recommended):** Click **Session** > **SSH** > Remote host: `grace.hprc.tamu.edu` > Check **Specify username** and enter your NetID.
  * **PuTTY:** Connection type: `SSH`, Port: `22`, Host Name: `grace.hprc.tamu.edu`.

*Note: For off-campus access, connect to the Texas A&M VPN prior to starting your SSH session.*

---

## Navigating Grace & Storage Quotas

| Directory | Path / Environment Variable | Storage Limit | File Limit | Backed Up? | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Home** | `$HOME` (`/home/NetID`) | 10 GB (Fixed) | 10,000 | Yes (Nightly) | Scripts, small files, config |
| **Scratch** | `$SCRATCH` (`/scratch/user/NetID`) | 1 TB (Expandable) | 250,000 | **NO** | General computation & datasets |

### Navigation Commands
```bash
# Navigate to Home directory
cd $HOME

# Navigate to Scratch directory
cd $SCRATCH

# Check current storage quota usage
showquota
```

---

## File Transfer

### Methods
* **GUI Tools:** MobaXterm SFTP, WinSCP, FileZilla (Note: FileZilla may experience Duo 2FA issues).
* **Command Line (Fastest):** `scp`, `rsync`.
* **Large File Transfers (> several GB):** Use `rsync` or **Globus**.

### Example `rsync` Command
```bash
rsync -av [-z] localdir/ username@remotesystem:/path/to/remotedir/
```

---

## Managing Project Accounts

View active accounts and configure your default account for Service Unit (SU) billing:
```bash
myproject
```

---

## Software & Hierarchical Modules

Software on Grace is managed through hierarchical environment modules.

Finding Software

Command-line options for finding available software on our clusters include:

module avail: Lists modules available to load based on currently loaded dependencies.

module spider: Searches all module pathways for a specific keyword or package (most up-to-date).

fmq: Fast Module Query — a faster alternative to module spider.

mla: Lists/searches software installed as modules across HPRC clusters.

Note: The fmq command is a fast alternative to module spider, but module spider will have the most up-to-date module information.

Examples Using fmq

# Search for available Python modules
fmq Python

# Query information about a specific module version
fmq Python/3.8.2-GCCcore-9.3.0


```bash
# Search for software across all installed modules
mla keyword

# Search for module installation pathways
module spider keyword

# View specific module prerequisites (e.g., Perl/5.32.0)
module spider Perl/5.32.0

# Load base dependency and target module
module load GCCcore/10.2.0 Perl/5.32.0

# List currently loaded modules
module list

# See additional modules compatible with currently loaded dependencies
module avail

# Unload all loaded modules
module purge
```

---

## Batch System (Slurm)

All resource-intensive workloads must run on compute nodes via **Slurm**. Running computational jobs on login nodes is strictly prohibited.

* **Typical Grace Compute Node Hardware:** 48 cores, 384 GB usable RAM.

### Sample Job Script (`MyJob.slurm`)
```bash
#!/bin/bash
## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample1     # Job name
#SBATCH --time=01:30:00            # Wall clock limit (HH:MM:SS)
#SBATCH --ntasks=1                 # Total tasks
#SBATCH --ntasks-per-node=1        # Tasks per node
#SBATCH --mem=2560M                # RAM per node (2.5GB)
#SBATCH --output=Example1Out.%j    # Output/Error log (%j expands to jobID)

# First Executable Line
module load GCC/10.2.0
./my_executable
```

> **Windows File Conversion Notice:** If your script was edited on Windows, strip carriage return characters before submitting:
> ```bash
> dos2unix MyJob.slurm
> ```

### Submitting & Monitoring Jobs

```bash
# Submit a job
sbatch MyJob.slurm

# Check status of all your jobs
squeue -u $USER

# Check status of a specific job ID
squeue --job <JOBID>

# Cancel a running or queued job
scancel <JOBID>
```

### Automatic Submissions (`tamubatch`)
`tamubatch` allows job submission without manually writing Slurm scripts by supplying executable commands directly in a text file.

---

## Graphical User Interfaces (GUIs)

1. **HPRC Open OnDemand Portal:** Web-based interface requiring campus network or VPN connection. Supports direct interactive GUI apps and VNC desktop sessions.
2. **X11 Forwarding:** Connect via SSH with X11 forwarding enabled to run light GUI software on login nodes in accordance with the login node usage policy.

---

## Workflow Example: Deep Learning (TensorFlow / PyTorch)

### 1. Environment Setup

```bash
# Clear existing modules
module purge

# Load modules required for TensorFlow / CUDA
module load GCCcore/9.3.0 GCC/9.3.0 Python/3.8.2 CUDAcore/11.0.2 CUDA/11.0.2 cuDNN/8.0.5.39-CUDA-11.0.2

# Save module environment for future sessions
module save dl

# Restore saved module environment
module restore dl

# Create and activate Python virtual environment in Scratch
cd $SCRATCH
python -m venv dlvenv
source dlvenv/bin/activate

# Upgrade pip
pip install -U pip
```

### 2. Package Installation & GPU Verification

```bash
# Install TensorFlow
pip install tensorflow
python -c "import tensorflow as tf; print(tf.test.gpu_device_name())"

# Install PyTorch
pip install torch torchvision
python -c "import torch; print(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))"
```

### 3. Deep Learning Slurm Script (`dl.slurm`)

```bash
#!/bin/bash
## NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample4       # Job name
#SBATCH --time=00:30:00              # Wall clock limit
#SBATCH --ntasks=1                   # Total tasks
#SBATCH --mem=2560M                  # RAM per task
#SBATCH --output=Example4Out.%j      # Output file
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --partition=gpu              # GPU partition

# Load required modules
module load GCCcore/9.3.0 GCC/9.3.0 Python/3.8.2 CUDAcore/11.0.2 CUDA/11.0.2 cuDNN/8.0.5.39-CUDA-11.0.2

# Activate virtual environment
source $SCRATCH/dlvenv/bin/activate

# Execute python script
cd $SCRATCH/mywonderfulproject
python TuringTest.py
```

Submit GPU job:
```bash
sbatch dl.slurm
```

## tamubatch Utility Detailed Guide

`tamubatch` is an automated job submission tool on Grace that simplifies submitting Slurm jobs without manually writing `#SBATCH` header directives. Users supply a standard script file containing executable commands, and `tamubatch` automatically formats and submits the batch job using either default parameters or user-specified flags.

### Command Synopsis & Options

```bash
tamubatch [JOB_FILE] [OPTIONS]
```

| Flag | Short | Parameter | Description | Default |
| :--- | :--- | :--- | :--- | :--- |
| `--walltime` | `-W` | `<H:MM>` | Sets job walltime limit | `0:30` (30 minutes) |
| `--GPU` | `-gpu` | None | Directs job to run on GPU nodes | Disabled |
| `--cores` | `-n` | `<n>` | Total number of CPU cores requested | `1` core |
| `--cores-per-node` | `-R` | `<n>` | Cores per node (Max 48 per node on Grace) | Same as `-n` (up to 48) |
| `--total-memory` | `-M` | `<n>MB/G` | Total memory limit (must specify `MB` or `G`) | $4000\text{ MB} \times \text{cores}$ |
| `--project-account` | `-P` | `<Account>` | Specifies SU allocation account to charge | Default user account |
| `--extras` | `-x` | `"<flags>"` | Passes additional Slurm flags | None |
| `--command` | `-command` | `"<bash>"` | Appends bash commands to job file before running | None |
| `--download` | `-download` | None | Generates script file without submitting job | Off (submits directly) |
| `--help` | `-h` | None | Displays usage documentation and exits | N/A |

### Batch File Format

The input script is a standard text/bash script containing the commands you need executed on the cluster:

```bash
#!/bin/bash

echo "Hello"
ml purge
./my_script
```

### Usage Examples

#### 1. Default Job Submission
Submits `my_job_file` with default limits (1 core, 30 min walltime, 4 GB memory):
```bash
tamubatch my_job_file
```

#### 2. Custom Resources (Time, CPU Cores, Memory)
Submits a job requesting 1 hour walltime, 20 CPU cores on 1 node, and 50 GB total memory:
```bash
tamubatch my_job_file -W 1:00 -n 20 -R 20 -M 50G
```

#### 3. GPU Job with Extra Slurm Parameters
Requests a GPU node, 5 hours walltime, 40 cores (20 cores/node), 80 GB RAM, and attaches Slurm email notifications via the `-x` flag:
```bash
tamubatch my_job_file -gpu -W 5:00 -n 40 -R 20 -M 80G -x "--mail-type=ALL --mail-user=NetID@tamu.edu"
```

#### 4. Appending Inline Commands (Rapid Prototyping)
Appends inline bash commands to the end of `my_job_file` prior to submission:
```bash
tamubatch my_job_file -W 1:00 -n 20 -R 20 -M 50G -command "echo hello; cd /user/net-id/"
```