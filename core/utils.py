"""Utility functions for parallel processing and recursive attribute access.

Originally from snel_toolkit.utils. Extracted for standalone use.
"""

import functools
import multiprocessing


def rgetattr(obj, attr, *args):
    """A recursive drop-in replacement for getattr that handles dotted attr strings.

    Parameters
    ----------
    obj : object
        The object to retrieve the attribute from.
    attr : str
        A dotted attribute string (e.g., 'spikes.columns').
    *args
        Default value if the attribute is not found.

    Returns
    -------
    object
        The value of the nested attribute.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def fun(f, q_in, q_out):
    """Worker function for parmap multiprocessing."""
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """Parallel map that works with functions inside class methods.

    Unlike multiprocessing.Pool.map, this implementation uses
    Queue-based communication and supports closures/lambdas.

    Parameters
    ----------
    f : callable
        Function to apply to each element.
    X : iterable
        Input data to map over.
    nprocs : int, optional
        Number of processes, by default uses all available CPUs.

    Returns
    -------
    list
        Results in the same order as the input.
    """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [
        multiprocessing.Process(target=fun, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
