#!/usr/bin/env bash

import os
import tensorflow as tf
from worker import WorkerFactory
from environment import create_env
from mathlib import log_uniform
from argparser import ArgParser
import numpy as np
import sys

def train():

    learning_rate = 0.0002

    ps_hosts = np.zeros(args.ps_hosts_num, dtype=object)
    worker_hosts = np.zeros(args.worker_hosts_num, dtype=object)
    port_num = args.st_port_num

    for i in range(args.ps_hosts_num):
        ps_hosts[i] = str(args.hostname) + ":" + str(port_num)
        port_num += 1

    for i in range(args.worker_hosts_num):
        worker_hosts[i] = str(args.hostname) + ":" + str(port_num)
        port_num += 1

    ps_hosts = list(ps_hosts)
    worker_hosts = list(worker_hosts)

    # Create a cluster from the parameter server and worker hosts
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task
    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)

    if args.job_name == "ps":
        server.join()

    elif args.job_name == "worker":
        device = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % args.task_index,
            cluster=cluster)

        with tf.device(device):

            tf.set_random_seed(args.task_index * 89)

            vars(args)['worker_name'] = 'worker_name'
            vars(args)['worker_index'] = 'worker_' + str(args.task_index)
            vars(args)['device'] = device
            vars(args)['task_index'] = args.task_index
            vars(args)['learning_rate'] = learning_rate
            local_env = create_env(**vars(args))
            vars(args)['env'] = local_env

            worker = WorkerFactory.create_worker(**vars(args))

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            global_step_ph = tf.placeholder(global_step.dtype, shape=global_step.get_shape())
            global_step_ops = global_step.assign(global_step_ph)
            init_op = tf.global_variables_initializer()
            saver = tf.train.Saver()

        sv = tf.train.Supervisor(is_chief=(args.task_index == 0),
                                 global_step=global_step,
                                 logdir=str(os.getcwd()) + '/tmp/' + vars(args)['algo_name'] + '/' +str(vars(args)['learning_rate']) + '/'+ str(args.task_index) + '/',
                                 saver=saver,
                                 init_op=init_op)

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % args.task_index])

        with sv.managed_session(server.target, config=config) as sess:
            while True:
                if sess.run([global_step])[0] > vars(args)['max_master_time_step']:
                    break

                diff_global_t = worker.work(sess, current_master_timestep=sess.run([global_step])[0],)
                sess.run(global_step_ops, {global_step_ph: sess.run([global_step])[0] + diff_global_t})
        sv.stop()

def main(_):
    os.system("rm -rf " + str(os.getcwd())+'/checkpoints/*')
    train()

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    tf.app.run(main=main, argv=[sys.argv[0]])
