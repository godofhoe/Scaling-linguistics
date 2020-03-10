from warnings import warn

from collections import defaultdict
from os.path import splitext
from contextlib import contextmanager
try:
    from pathlib import Path
except ImportError:
    # Use Path to indicate if pathlib exists (like numpy does)
    Path = None

import networkx as nx
from decorator import decorator

def not_implemented_for(*graph_types):
    """Decorator to mark algorithms as not implemented

    Parameters
    ----------
    graph_types : container of strings
        Entries must be one of 'directed','undirected', 'multigraph', 'graph'.

    Returns
    -------
    _require : function
        The decorated function.

    Raises
    ------
    NetworkXNotImplemented
    If any of the packages cannot be imported

    Notes
    -----
    Multiple types are joined logically with "and".
    For "or" use multiple @not_implemented_for() lines.

    Examples
    --------
    Decorate functions like this::

       @not_implemnted_for('directed')
       def sp_function(G):
           pass

       @not_implemnted_for('directed','multigraph')
       def sp_np_function(G):
           pass
    """
    @decorator
    def _not_implemented_for(not_implement_for_func, *args, **kwargs):
        graph = args[0]
        terms = {'directed': graph.is_directed(),
                 'undirected': not graph.is_directed(),
                 'multigraph': graph.is_multigraph(),
                 'graph': not graph.is_multigraph()}
        match = True
        try:
            for t in graph_types:
                match = match and terms[t]
        except KeyError:
            raise KeyError('use one or more of ',
                           'directed, undirected, multigraph, graph')
        if match:
            msg = 'not implemented for %s type' % ' '.join(graph_types)
            raise nx.NetworkXNotImplemented(msg)
        else:
            return not_implement_for_func(*args, **kwargs)
    return _not_implemented_for

def py_random_state(random_state_index):
    """Decorator to generate a random.Random instance (or equiv).

    Argument position `random_state_index` processed by create_py_random_state.
    The result is either a random.Random instance, or numpy.random.RandomState
    instance with additional attributes to mimic basic methods of Random.

    Parameters
    ----------
    random_state_index : int
        Location of the random_state argument in args that is to be used to
        generate the numpy.random.RandomState instance. Even if the argument is
        a named positional argument (with a default value), you must specify
        its index as a positional argument.

    Returns
    -------
    _random_state : function
        Function whose random_state keyword argument is a RandomState instance.

    Examples
    --------
    Decorate functions like this::

       @py_random_state(0)
       def random_float(random_state=None):
           return random_state.rand()

       @py_random_state(1)
       def random_array(dims, random_state=1):
           return random_state.rand(*dims)

    See Also
    --------
    np_random_state
    """
    @decorator
    def _random_state(func, *args, **kwargs):
        # Parse the decorator arguments.
        try:
            random_state_arg = args[random_state_index]
        except TypeError:
            raise nx.NetworkXError("random_state_index must be an integer")
        except IndexError:
            raise nx.NetworkXError("random_state_index is incorrect")

        # Create a numpy.random.RandomState instance
        random_state = create_py_random_state(random_state_arg)

        # args is a tuple, so we must convert to list before modifying it.
        new_args = list(args)
        new_args[random_state_index] = random_state
        return func(*new_args, **kwargs)
    return _random_state


def create_py_random_state(random_state=None):
    """Returns a random.Random instance depending on input.

    Parameters
    ----------
    random_state : int or random number generator or None (default=None)
        If int, return a random.Random instance set with seed=int.
        if random.Random instance, return it.
        if None or the `random` package, return the global random number
        generator used by `random`.
        if np.random package, return the global numpy random number
        generator wrapped in a PythonRandomInterface class.
        if np.random.RandomState instance, return it wrapped in
        PythonRandomInterface
        if a PythonRandomInterface instance, return it
    """
    import random
    try:
        import numpy as np
        if random_state is np.random:
            return PythonRandomInterface(np.random.mtrand._rand)
        if isinstance(random_state, np.random.RandomState):
            return PythonRandomInterface(random_state)
        if isinstance(random_state, PythonRandomInterface):
            return random_state
        has_numpy = True
    except ImportError:
        has_numpy = False

    if random_state is None or random_state is random:
        return random._inst
    if isinstance(random_state, random.Random):
        return random_state
    if isinstance(random_state, int):
        return random.Random(random_state)
    msg = '%r cannot be used to generate a random.Random instance'
    raise ValueError(msg % random_state)

def cumulative_distribution(distribution):
    """Return normalized cumulative distribution from discrete distribution."""

    cdf=[]
    cdf.append(0.0)
    psum=float(sum(distribution))
    for i in range(0,len(distribution)):
        cdf.append(cdf[i]+distribution[i]/psum)
    return cdf

@py_random_state(3)
def discrete_sequence(n, distribution=None, cdistribution=None, seed=None):
    """
    Return sample sequence of length n from a given discrete distribution
    or discrete cumulative distribution.

    One of the following must be specified.

    distribution = histogram of values, will be normalized

    cdistribution = normalized discrete cumulative distribution

    """
    import bisect

    if cdistribution is not None:
        cdf = cdistribution
    elif distribution is not None:
        cdf = cumulative_distribution(distribution)
    else:
        raise nx.NetworkXError(
            "discrete_sequence: distribution or cdistribution missing")

    # get a uniform random number
    inputseq = [seed.random() for i in range(n)]

    # choose from CDF
    seq = [bisect.bisect_left(cdf, s) - 1 for s in inputseq]
    return seq   
 
class PythonRandomInterface(object):
    try:
        def __init__(self, rng=None):
            import numpy
            if rng is None:
                self._rng = numpy.random.mtrand._rand
            self._rng = rng
    except ImportError:
        msg = 'numpy not found, only random.random available.'
        warnings.warn(msg, ImportWarning)

    def random(self):
        return self._rng.random_sample()

    def uniform(self, a, b):
        return a + (b - a) * self._rng.random_sample()

    def randrange(self, a, b=None):
        return self._rng.randint(a, b)

    def choice(self, seq):
        return seq[self._rng.randint(0, len(seq))]

    def gauss(self, mu, sigma):
        return self._rng.normal(mu, sigma)

    def shuffle(self, seq):
        return self._rng.shuffle(seq)

#    Some methods don't match API for numpy RandomState.
#    Commented out versions are not used by NetworkX

    def sample(self, seq, k):
        return self._rng.choice(list(seq), size=(k,), replace=False)

    def randint(self, a, b):
        return self._rng.randint(a, b + 1)

#    exponential as expovariate with 1/argument,
    def expovariate(self, scale):
        return self._rng.exponential(1/scale)

#    pareto as paretovariate with 1/argument,
    def paretovariate(self, shape):
        return self._rng.pareto(shape)

#    weibull as weibullvariate multiplied by beta,
#    def weibullvariate(self, alpha, beta):
#        return self._rng.weibull(alpha) * beta
#
#    def triangular(self, low, high, mode):
#        return self._rng.triangular(low, mode, high)
#
#    def choices(self, seq, weights=None, cum_weights=None, k=1):
#        return self._rng.choice(seq