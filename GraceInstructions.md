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