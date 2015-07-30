# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:54:51 2015

@author: Jan
"""

import multiprocessing as mp
from time import sleep
import numpy as np
import sys


def worker(comparers, pipe):
    print '    ', mp.current_process().name, 'starting!'
    sys.stdout.flush()
    for value, comparer_id in iter(pipe.recv, 'STOP'):
        print '    {} processes value: {}'.format(mp.current_process().name, value)
        sys.stdout.flush()
        result = '    {} result: {}'.format(mp.current_process().name,
                                            comparers[comparer_id](value))
        pipe.send(result)
    print '    ', mp.current_process().name, 'exiting!'
    sys.stdout.flush()


class Comparer(object):

    def __init__(self, reference):
        self.reference = reference

    def __call__(self, value):
        outcome = self.reference == value
        return 'Comparer({}) compares with {}: {}'.format(self.reference, value, outcome)


class Main(object):

    def __call__(self):
        nprocs = 4
        nvalues = 17
        values = range(nvalues)
        comparers = np.asarray([Comparer(i) for i in range(nvalues)])
        print 'PARENT Create communication pipes'
        pipes = [mp.Pipe() for i in range(nprocs)]
        sleep(1)
        print 'PARENT Setup a list of processes that we want to run'
        processes = []
        proc_ids = np.asarray([i % nprocs for i in range(nvalues)])
        for proc_id in range(nprocs):
            selection = comparers[np.where(proc_ids == proc_id, True, False)]
            processes.append(mp.Process(name='WORKER {}'.format(proc_id), target=worker,
                                        args=(selection, pipes[proc_id][1])))
        sleep(1)
        print 'PARENT Run processes'
        for p in processes:
            p.start()
        sleep(1)
        print 'PARENT Distribute work'
        for i in range(nvalues):
            proc_id = i % nprocs
            comparer_id = i // nprocs
            pipes[proc_id][0].send((values[i], comparer_id))
        sleep(1)
        print 'PARENT Get process results from the pipes'
        results = []
        for i in range(nvalues):
            proc_id = i % nprocs
            results.append(pipes[proc_id][0].recv())
        sleep(1)
        print 'PARENT Print results'
        for result in results:
            print result
        sleep(1)
        print 'PARENT Finalize processes'
        for i in range(nprocs):
            pipes[i][0].send('STOP')
        sleep(1)
        print 'PARENT Exit the completed processes'
        for p in processes:
            p.join()


if __name__ == '__main__':
    mp.freeze_support()
    Main()()
