# Team-project-2023
You can update this file if you think you find anything worth to say for our team project.

# How to run the code
1. If you use your own PC, click 'console' in Pycharm.

2. Type 'python FILE-NAME.py  --gpu-id 0 --IPC 16 --dataset-config-file configs/datasets/domain_ipc_pacs.yaml --config-file configs/xded_default.yaml --trainer XDED --remark XDED_UniStyle12 MODEL.BACKBONE.NAME resnet18_UniStyle_12' in cmd to run.

3. To know more about these arguments, you can check the file 'XDED-main/options.py'.

# About run the epoch
You can find a file 'XDED-main/dassl/engine/trainer.py'. In this file, author defined how to train the model.

There is a function 'def run_epoch(self)' under the class 'class TrainerX(SimpleTrainer)'. In this function, you can find the function who calculate 'loss' and 'lr'.

In the file 'XDED-main/dassl/engine/xded.py', function 'def forward_backward(self, batch)' (under the class 'class XDED(TrainerX)') calculate the value of the 'loss'. (vanilla.py also has this function)

# Knowledge Distillation (a method used in the paper we choose)
You can search 'Knowledge Distillation' on google to know more about it.

This paper use self-distillation. i.e. the model is both 'teacher' and 'student'.

In my opinion, for Single-source domain generalization, it means that the train set is domain A, but the test set is domain B. (For example, we use 'photo' to train the model, then use 'cartoon' to test the accuracy. If it has a good performance， means the model is generalized.) 
Of course, we can also use 1 domain to train, and test on 2 or 3 domains, or train with 2 to 3 domains, and test on the rest domains.
