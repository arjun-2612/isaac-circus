# Husky Beta RL Based Locomotion
In this directory, you will find all files related to the simulation of Husky-b locomotion. They have been listed below:
| File Name | Description |
| ------------ | ----------- |
| [```RSL-RL PPO Config```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/blob/main/source/sslab_extensions/tasks/locomotion/husky_beta/agents/rsl_rl_ppo_cfg.py) | contains all hyperparameters used in the PPO runner, and Actor-Critic configuration for training |
| [```MDP```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/tree/main/source/sslab_extensions/tasks/locomotion/husky_beta/mdp) | contains the curriculum, domain randomization, and rewards/penalties used in trainings |
| [```Flat Terrain Config```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/blob/main/source/sslab_extensions/tasks/locomotion/husky_beta/flat_env_cfg.py) | contains the manager-based unification of flat terrain task setup |
| [```Stairs-Waves Terrain Config```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/blob/main/source/sslab_extensions/tasks/locomotion/husky_beta/stairs_waves_env_cfg.py) | contains the manager-based unificiation of stairs and waves terrain task setup |
| [```Multimodal Terrain Config```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/blob/main/source/sslab_extensions/tasks/locomotion/husky_beta/multimodal_env_cfg.py) | contains the manager-based unification of multimodal terrain task setup |
| [```Task Initialization```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/blob/main/source/sslab_extensions/tasks/locomotion/husky_beta/__init__.py) | contains the Gymnasium task registration and entry points information |

The asset file used in training can be found [```here```](https://github.com/SS-Lab-at-NU/sslab_isaaclab/blob/main/source/sslab_extensions/assets/husky_beta.py)

More information on the specifics of the task and terrain setup can be found in the RL tutorial PowerPoint file on SharePoint, linked [```here```](https://northeastern-my.sharepoint.com/:p:/r/personal/aramez_northeastern_edu/_layouts/15/Doc.aspx?sourcedoc=%7BEB0D3508-21AD-435F-BF8F-222CB0E2EED2%7D&file=SSLab_RL_tutorial.pptx&action=edit&mobileredirect=true) from slide 40. 
