# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:54:51 2015

@author: Jan
"""

import threading
import multiprocessing
import numpy as np
from jutil.taketime import TakeTime


def func(n):
    if n < 2:
        return []
    factors = []
    p = 2

    while True:
        if n == 1:
            return factors

        r = n % p
        if r == 0:
            factors.append(p)
            n = n / p
        elif p * p >= n:
            factors.append(n)
            return factors
        elif p > 2:
            # Advance in steps of 2 over odd numbers
            p += 2
        else:
            # If p == 2, get to 3
            p += 1
    assert False, "unreachable"


def serial_func(nums):
    return {n: func(n) for n in nums}


def threaded_worker(nums, outdict):
        """ The worker function, invoked in a thread. 'nums' is a
            list of numbers to factor. The results are placed in
            outdict.
        """
        for n in nums:
            outdict[n] = func(n)


def threaded_func(nums, nthreads):
    # Each thread will get 'chunksize' nums and its own output dict
    chunksize = int(np.ceil(len(nums) / float(nthreads)))
    threads = []
    outs = [{} for i in range(nthreads)]
    for i in range(nthreads):
        # Create each thread, passing it its chunk of numbers to factor and output dict.
        t = threading.Thread(target=threaded_worker,
                             args=(nums[chunksize * i:chunksize * (i + 1)], outs[i]))
        threads.append(t)
        t.start()
    # Wait for all threads to finish
    for t in threads:
        t.join()
    # Merge all partial output dicts into a single dict and return it
    return {k: v for out_d in outs for k, v in out_d.iteritems()}


def multiproc_worker(nums, out_q):
    """ The worker function, invoked in a process. 'nums' is a
        list of numbers to factor. The results are placed in
        a dictionary that's pushed to a queue.
    """
    outdict = {}
    for n in nums:
        outdict[n] = func(n)
    out_q.put(outdict)


def multiproc_func(nums, nprocs):
    # Each process will get 'chunksize' nums and a queue to put his out dict into
    out_q = multiprocessing.Queue()
    chunksize = int(np.ceil(len(nums) / float(nprocs)))
    procs = []
    for i in range(nprocs):
        p = multiprocessing.Process(target=multiproc_worker,
                                    args=(nums[chunksize * i:chunksize * (i + 1)], out_q))
        procs.append(p)
        p.start()
    # Collect all results into a single result dict. We know how many dicts with results to expect.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    return resultdict


if __name__ == '__main__':
    multiprocessing.freeze_support()

    input_array = np.arange(1E6)

    with TakeTime('serial') as timer:
        serial_func(input_array)
        print 'Serial Time:', timer.dt

    with TakeTime('threaded') as timer:
        threaded_func(input_array, 4)
        print 'Threaded Time:', timer.dt

    with TakeTime('multiprocessing') as timer:
        multiproc_func(input_array, 4)
        print 'Multiprocessing Time:', timer.dt
