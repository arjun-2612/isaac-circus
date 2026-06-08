# IsaacLab Circus
This repository contains the following IsaacLab simulation files for robots from my learning based control work in graduate school:

## Authors
Arjun Viswanathan 

# Important Things to Know
These are the core rules to follow with this repository. The git tree or the installs will most likely break if you do not follow these rules, and it will be a headache to restore it back. So please, take the time to read this section. This section will be expanded as more Do's and Don'ts are encountered. 

1. Set everything up locally on your laptop/PC. Do NOT use Explorer cluster to do any development. It is solely used for downloading runs and training policies ONLY. 
2. Do NOT add USD files to the commits. They are too big and need git LFS, which I have no configured yet. Only have the URDF and meshes. The USD should be moved outside the repo to wherever you want for now
3. Please keep all changes to SLURM resources or training/playing to your YAML file ONLY. Modifying the main shell script WILL break things for others
4. Make a branch from main for yourself and work in that branch. Do not make commits to main. You will need to make PRs for your changes to reflect on main
5. The docker setup for Explorer will use up your /home/$USER storage. To avoid hitting storage quote, run ```rm -rf ~/.apptainer``` on the cluster every now and then. You can check usage via ```du -sh /home/$USER/.apptainer```
6. If you are using a regular laptop not capable of running Isaac, that is totally fine. You don't need to install anything. Just set up docker, W&B, and the cluster login. You can just use the cluster to run trainings

# Cloning
Since this repository has submodules, make sure you clone recursively. In VS Code, when you do ```ctrl+shift+p``` and type ```clone``` it will show an option ```Clone (Recursive)```. Make sure you select that and paste the link to this repo. Ideally drop this repo into your ```Documents``` directory.

For using Git CLI, in the terminal type:
```
git clone --recursive <git_repo_link>
```

If you already have a version of this git repo and there have been updates to submodules, then from the root of the repo, use:
```
git submodule update --init --recursive
```

# IsaacSim 4.5.0 and IsaacLab 2.1.0 installation
Only follow this section if you are on a local workstation that is CAPABLE OF RUNNING ISAACSIM/LAB (RTX GPU 2070 or higher with 8+ GB VRAM, 32+ GB RAM, decent CPU, and at least 50 GB storage space).

First, you will need to make a conda environment wherever you want to install IsaacSim/Lab and the extensions in this repo. If you don't already have conda installed, use this [```link```](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer) to install. Then run this command in the terminal
```
conda create -n isaac450 python=3.10 
```

Consult this [```link```](https://isaac-sim.github.io/IsaacLab/release/2.1.0/source/setup/installation/pip_installation.html) on setting up IsaacSim and IsaacLab from the [```official repository```](https://github.com/isaac-sim/IsaacLab/tree/release/2.1.0).

Once you have followed the link and installed IsaacSim/Lab, you need to install the extensions in this repo. Use the [```install_extensions.sh```](https://github.com/arjun-2612/isaac-circus/blob/main/install_extensions.sh) shell script as below from the main directory of the repo:
```
./install_extensions.sh
```

Our trainings use Weights and Biases (WandB) as the logger. You will need to enable this. Please request access to the SS Lab WandB project from one of the maintainers. After you have access, you need to find the WandB API key for the project. Then go to your terminal and do this: 
```
wandb login
```
It will then ask you to enter the API key you copied.

Then, you can start training locally!

For using the cluster or another machine via SSH, you need to save a file locally with the key on it. The simplest way is shown below 
```
nano ~/.wandb_api_key_file
```
Then paste the key and save (CTRL+O) and exit (CTRL+X). This file will be saved to your home directory locally. Then, the main script should take it from there. 

# Cluster Setup - Docker + SSH
If you are using the cluster, you need to first set up an account with Docker, and make a repository for storing your images. Choose any name, but make sure it is consistent in the [```YAML```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/simulation/config/go2.yaml#L43) files. Here is an [```example```](https://hub.docker.com/repository/docker/arjunviswanathan/sslab-isaaclab/general) Docker repository. Make sure you log into your account on your host machine
```
docker login -u <username>
```

Next up, you need to follow the [```cluster docs```](https://rc-docs.northeastern.edu/en/explorer-main/connectingtocluster/linux.html#passwordless-ssh-in-linux) to allow passwordless access into the cluster. This is only available in Linux. 

Once both are set up, you are ready to start using the cluster!

# Training 
To run trainings for each robot, use the [```isaac_circus_main.sh```](https://github.com/arjun-2612/isaac-circus/blob/main/isaac_circus_main.sh) shell script by typing the following commands in the terminal respectively from the main directory of this repo. You will pass in the robot name as a parameter to the scripts. The options to choose from are whatever robot names exist in the [```assets```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/assets) folder.

For a native workstation, use:
```
./isaac_circus_main.sh <robot_name>
```
For Explorer cluster, use:
```
./isaac_circus_main.sh <robot_name> cluster
```

Note that now you will need to add all the parameters you want into the corresponding YAML file. These files are located [```here```](https://github.com/arjun-2612/isaac-circus/tree/main/source/circus_extensions/simulation/config). Just specify the task name, which can be found in the respective init file. For example, to specify a task in Husky-b, I will copy the id field from [```__init__.py```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/tasks/locomotion/husky_beta/__init__.py) file in that directory. 

# Exporting
If you trained in the Explorer cluster, you need to create the ONNX file. You can do this by editing the YAML file for your robot like so
1. Set [```mode: play```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/simulation/config/go2.yaml#L3) and [```task.name: <play_version_of_task>```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/simulation/config/go2.yaml#L6) so it generates the ```exported``` folder in your log folder
2. Set the [```image tag```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/simulation/config/go2.yaml#L46) to the one on your SIF image so it knows which container to run the task in.
3. Then run this command:
```
./isaac_circus_main.sh <robot_name> cluster
```
If you trained locally on your own workstation, use
```
./isaac_circus_main.sh <robot_name>
```
after making the same changes to your YAML file.

# Playing in IsaacSim
To play the trained policies in IsaacSim locally, you will first download the log folder in ```logs/rsl_rl/<robot_name>_rl/<folder_name>``` corresponding to the time signature of your run from the cluster into the same location locally (this is very important). Then, edit your [```run_folder name```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/simulation/isaacsim_scripts/go2/extensions/go2_example.py#L48) to the folder you want to simulate (in the example.py file for your robot).

Use the main script again, set ```mode: sim``` in the corresponding YAML file. This will copy over your simulation files located [```here```](https://github.com/arjun-2612/isaac-circus/tree/main/source/circus_extensions/simulation/isaacsim_scripts) for your robot into Isaacsim's user examples folder. Make sure you comment in the lines you need in the [```__init__.py```](https://github.com/arjun-2612/isaac-circus/blob/main/source/circus_extensions/simulation/isaacsim_scripts/__init__.py) file so it activates your extension. 

# Northeastern University Clusters
You can also choose how much RAM you want, what GPU to use, how many CPUs, the logging directory, and the specific training related items you wish to use. Consult the [```Northeastern University Discovery Cluster Docs```](https://rc-docs.northeastern.edu/en/latest/) and [```Northeastern University Explorer Cluster Docs```](https://rc-docs.northeastern.edu/en/explorer-main/index.html) on how to use sbatch scripts and choose resources. 

NOTE: This will need to input into the YAML file for your corresponding robot. Do NOT modify the main script as this will need to be used by everyone using the repo. Only modify your OWN YAML file!!!
