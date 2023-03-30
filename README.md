# Team-project-2023
paper: https://paperswithcode.com/paper/cross-domain-ensemble-distillation-for-domain

You can update this file if you think you find anything worth to say for our team project.

If you can not understand the code, https://github.com/KaiyangZhou/Dassl.pytorch may help you

The net used in the paper (resnet18_UniStyle_12) is registered in the file 'XDED-main/dassl/modeling/backbone/whiten_resnet.py'

# How to run the code
1. If you use your own PC, click 'console' in Pycharm.

2. Type 
```c
python FILE-NAME.py  --gpu-id 0 --IPC 16 --dataset-config-file configs/datasets/domain_ipc_pacs.yaml --config-file configs/xded_default.yaml --trainer XDED --remark XDED_UniStyle12 MODEL.BACKBONE.NAME resnet18_UniStyle_12
```
  
  in cmd to run.

3. To know more about these arguments, you can check the file 'XDED-main/options.py'.

# How to run the code on HPC
1. Require a 8G RAM and a GPU
```c 
srun --account=dcs-res --partition=dcs-gpu --mem=8G --nodes=1 --gpus-per-node=1 --pty bash //require GPU from dcs with 8G RAM
```
2. Load conda module and cuDNN module
```c
module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b
```
3. Create a conda environment ('environment.yaml' is a file in XDED-main)
```c
conda env create --file XDED-main/environment.yaml
```
4. Active the environment
```c
source activate xded
```
5. Run the code just like on your PC.


# About run the epoch
You can find a file 'XDED-main/dassl/engine/trainer.py'. In this file, author defined how to train the model.

There is a function 'def run_epoch(self)' under the class 'class TrainerX(SimpleTrainer)'. In this function, you can find the function who calculate 'loss' and 'lr'.

In the file 'XDED-main/dassl/engine/xded.py', function 'def forward_backward(self, batch)' (under the class 'class XDED(TrainerX)') calculate the value of the 'loss'. (vanilla.py also has this function)

# Knowledge Distillation (a method used in the paper we choose)
You can search 'Knowledge Distillation' on google to know more about it.

This paper use self-distillation. i.e. the model is both 'teacher' and 'student'.

In my opinion, for Single-source domain generalization, it means that the train set is domain A, but the test set is domain B. (For example, we use 'photo' to train the model, then use 'cartoon' to test the accuracy. If it has a good performance， means the model is generalized.) 
Of course, we can also use 1 domain to train, and test on 2 or 3 domains, or train with 2 to 3 domains, and test on the rest domains.
