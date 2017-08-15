
import argparse
import os

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=None)
        self._add_arguments()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):
        self.parser.add_argument("--hostname", type=str, default="localhost", help="Hostname")
        self.parser.add_argument("--st-port-num", type=int, default=2222, help="Starting port number for processes")
        self.parser.add_argument("--job-name", type=str, default="worker", help="'One of ps' or 'worker'")
        self.parser.add_argument("--task-index", type=int, default=0, help="Task index within a job")
        self.parser.add_argument("--ps-hosts-num", type=int, default=1, help="The Number of Parameter Servers")
        self.parser.add_argument("--worker-hosts-num", type=int, default=1, help="The Number of Workers")

        self.parser.add_argument('--algo-name', default="a3c", help='Name of algorithm. For list, see README')
        self.parser.add_argument('--log-dir', default=os.getcwd() + "/tmp", help='Log directory path')
        self.parser.add_argument('--env-id', default="PongNoFrameskip-v4", help='Environment id') 
        self.parser.add_argument('--max-bootstrap-length', default=20, type=int, help='Max length of trajectory \
                                                                                 before bootstrapping')
        self.parser.add_argument('--max-master-time-step', default=999999999999999, type=int,
                            help='Max number of time steps to train')
        self.parser.add_argument('--max-clock-limit', default=0, type=float, help='Max clock limit to train')
        self.parser.add_argument('--anneal-learning-rate', action='store_true',
                            help='Flag to whether to anneal learning rate or not')
        self.parser.add_argument('--anneal-by-clock', action='store_true', help='Flag to anneal learning rate by clock time')
        self.parser.add_argument('--use-gpu', action='store_true', help='Flag to use gpu')

        def conv_layer_type(inpt):
            try:
                print(inpt)
                tup = eval(inpt)
                return tup
            except:
                raise argparse.ArgumentTypeError("Type in a list of 3-valued tuples e.g. [(16, 8, 4), (32, 4, 2)]\
                                                 where first value: # of filters, second value: 1-dim size of squared filter, \
                                                 third value: stride value")

        self.parser.add_argument('--convs', nargs='*', default=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],  #(32, 8, 4), (64, 4, 2), (64, 3, 1)  (16, 8, 4), (32, 4, 2)
                            help="Convolutional layer specification", type=conv_layer_type)
        self.parser.add_argument('--hiddens', nargs='*', type=int, default=[512],   # 256
                            help="Hidden layer specification: Type in a list of integers e.g. [256 256] where each element\
                                                 denotes the hidden layer node sizes in order given")
        self.parser.add_argument('--replay-buffer-size', default=1000000, type=int, help='Replay memory size')
        self.parser.add_argument('--exploration-fraction', default=0.1, type=float,
                            help='Exploration fraction, after which final eps is used')
        self.parser.add_argument('--exploration-final-eps', default=0.05, type=float,
                            help='Exploration afinal eps after exploration fraction * max time step.')
        self.parser.add_argument('--replay-start-size', default=50000, type=int,
                            help='random policy timesteps before actual learning begins')
        self.parser.add_argument('--train-update-freq', default=5, type=int,
                            help='number of actions between successive SGD updates')  #4
        self.parser.add_argument('--minibatch-size', default=32, type=int, help='minibatch size for SGD')
        self.parser.add_argument('--target-network-update-freq', default=10000, type=int,
                            help='target network update freq to stabilize learning')
