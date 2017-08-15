## A3C Implementation

This is an implementation of the Asynchronous advantage actor critic (A3C) algorithm using distributed tensorflow. 

The code structure extends from the DQN implementation - (https://github.com/woonsangcho/dqn-deep-q-network).

I referred to implementations from:

    openai (https://github.com/openai/universe-starter-agent)
    jaesik817 (https://github.com/jaesik817/a3c-distributed_tensorflow)

## Sample run script
```
ps_num=2
worker_num=4
python main_dist.py --ps-hosts-num=$ps_num --worker-hosts-num=$worker_num --job-name=ps --task-index=0 &
python main_dist.py --ps-hosts-num=$ps_num --worker-hosts-num=$worker_num --job-name=ps --task-index=1 &
python main_dist.py --ps-hosts-num=$ps_num --worker-hosts-num=$worker_num --job-name=worker --task-index=0 &
python main_dist.py --ps-hosts-num=$ps_num --worker-hosts-num=$worker_num --job-name=worker --task-index=1 &
python main_dist.py --ps-hosts-num=$ps_num --worker-hosts-num=$worker_num --job-name=worker --task-index=2 &
python main_dist.py --ps-hosts-num=$ps_num --worker-hosts-num=$worker_num --job-name=worker --task-index=3
```

## Sample result 
Using 2 parameter servers and 12 worker processes, 
![tensorboard visualization](https://github.com/woonsangcho/a3c/blob/master/a3c.jpg)
