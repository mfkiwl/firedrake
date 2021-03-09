"""

This module wraps `randomgen <https://pypi.org/project/randomgen/>`__
and enables users to generate a randomised :class:`.Function`
from a :class:`.FunctionSpace`.
This module inherits all attributes from `randomgen <https://pypi.org/project/randomgen/>`__.

"""
from mpi4py import MPI

import inspect
import warnings
import numpy as np
import numpy.random as randomgen

from firedrake.function import Function
from ufl import FunctionSpace

_deprecated_attributes = ['RandomGenerator', ]

__all__ = [name for name, _ in inspect.getmembers(randomgen, inspect.isclass)] + _deprecated_attributes


_known_attributes = ['beta', 'binomial', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'get_state', 'gumbel', 'hypergeometric', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power', 'rand', 'randint', 'randn', 'random', 'random_integers', 'random_sample', 'ranf', 'rayleigh', 'sample', 'seed', 'set_state', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf', 'Generator', 'RandomState', 'SeedSequence', 'MT19937', 'Philox', 'PCG64', 'SFC64', 'default_rng', 'BitGenerator']
_known_generator_attributes = ['beta', 'binomial', 'bit_generator', 'bytes', 'chisquare', 'choice', 'dirichlet', 'exponential', 'f', 'gamma', 'geometric', 'gumbel', 'hypergeometric', 'integers', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial', 'multivariate_hypergeometric', 'multivariate_normal', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'normal', 'pareto', 'permutation', 'poisson', 'power', 'random', 'rayleigh', 'shuffle', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'uniform', 'vonmises', 'wald', 'weibull', 'zipf']


def __getattr__(module_attr):

    # Reformat the original documentation
    def _reformat_doc(strng):

        # Reformat code examples
        st = ""
        flag = False
        strng = strng.replace('rs[i].jump(i)', '...     rs[i].jump(i)')
        strng = strng.replace('... ', '>>> ')
        for s in strng.splitlines():
            if flag and not ('>>>' in s or s.lstrip() == '' or s.lstrip()[0] == '#'):
                s = '>>> #' + s
            st += s + '\n'
            flag = '>>>' in s

        # Reformat the body
        strng = st
        st = ""
        for s in strng.splitlines():
            if 'from randomgen ' not in s:
                st += s.lstrip() + '\n'
        st = st.replace('randomgen', 'randomfunctiongen')
        st = st.replace('Parameters\n----------\n', '')
        st = st.replace('Returns\n-------\nout :', ':returns:')
        st = st.replace('Returns\n-------\nsamples :', ':returns:')
        st = st.replace('Returns\n-------\nZ :', ':returns:')
        st = st.replace('Raises\n-------\nValueError', '\n:raises ValueError:')
        st = st.replace('Raises\n------\nValueError', '\n:raises ValueError:')
        st = st.replace('Examples\n--------', '**Examples**\n')
        st = st.replace('Notes\n-----', '**Notes**\n')
        st = st.replace('See Also\n--------', '**See Also**\n')
        st = st.replace('References\n----------', '**References**')
        st = st.replace('\\P', 'P')
        st = st.replace('htm\n', 'html\n')
        st = st.replace('\n# ', '\n>>> # ')
        st = st.replace(':\n\n>>> ', '::\n\n    ')
        st = st.replace('.\n\n>>> ', '::\n\n    ')
        st = st.replace('\n\n>>> ', '::\n\n    ')
        st = st.replace('\n>>> ', '\n    ')

        # Convert some_par : -> :arg some_par:
        strng = st
        st = ""
        for s in strng.splitlines():
            if 'd0, d1, ..., dn :' in s:
                st += ':arg d0, d1, ..., dn' + s[16:] + '\n'
                continue
            elif ' ' in s and s.find(' ') != len(s)-1:
                n = s.find(' ')
                if s[n+1] == ':' and (n < len(s) - 2 and s[n+2] == ' '):
                    param_name = s[:n]
                    if param_name not in ('where', 'the', 'of', 'standard_normal') and 'scipy.stats' not in param_name and 'numpy.random' not in param_name:
                        st += ':arg ' + param_name + s[n+1:] + '\n'
                        continue
            st += s + '\n'

        # Remove redundant '\n' characters
        strng = st
        st = ""
        _in_block = False
        for s in strng.splitlines():
            if ':arg' in s or ':returns:' in s or '.. [' in s or '.. math::' in s:
                st += '\n' + s
                _in_block = True
                continue
            if _in_block:
                if s == '':
                    _in_block = False
                    st += '\n\n'
                else:
                    st += '. ' + s if s[0].isupper() and st[-1] != '.' else ' ' + s
            else:
                st += s + '\n'

        # Insert Firedrake wrapper doc and apply correct indentations
        strng = st
        st = ""
        sp = ' ' * 8
        for s in strng.splitlines():
            if "(d0, d1, ..., dn, dtype='d')" in s:
                name = s[:s.find('(')]
                st += sp + name + '(*args, **kwargs)\n\n'
                s = '*' + name + '* ' + s[len(name):]
                st += sp + s.replace('(', '(*').replace('d0, d1, ..., dn', 'V').replace(')', '*)') + '\n\n'
                st += sp + 'Generate a function :math:`f` = Function(V), internally call the original method *' + name + '* with given arguments, and return :math:`f`.\n\n'
                st += sp + ':arg V: :class:`.FunctionSpace`\n\n'
                st += sp + ':returns: :class:`.Function`\n\n'
                st += sp + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif 'size=None' in s:
                name = s[:s.find('(')]
                st += sp + name + '(*args, **kwargs)\n\n'
                s = '*' + name + '* ' + s[len(name):]
                st += sp + s.replace('(', '(*V, ').replace(', size=None', '').replace(')', '*)') + '\n\n'
                st += sp + 'Generate a :class:`.Function` f = Function(V), randomise it by calling the original method *' + name + '* (...) with given arguments, and return f.\n\n'
                st += sp + ':arg V: :class:`.FunctionSpace`\n\n'
                st += sp + ':returns: :class:`.Function`\n\n'
                st += sp + "The original documentation is found at `<https://numpy.org/doc/stable/reference/random/generated/numpy.random." + name + ".html>`__, which is reproduced below with appropriate changes.\n\n"
                st += sp + s.replace('(', '(*').replace(')', '*)') + '\n\n'
            elif '.. math::' in s:
                st += '\n' + sp + s + '\n\n'
            else:
                st += sp + s + '\n'

        return st
    if module_attr == 'Generator':
        _Base = getattr(randomgen, module_attr)
        _dict = {}
        _dict["__doc__"] = ("\n"
                            "    Container for the Basic Random Number Generators.\n"
                            "\n"
                            "    The original documentation is found at `<https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`__, which is reproduced below with appropriate changes.\n"
                            "\n"
                            "    Users can pass to many of the available distribution methods\n"
                            "    a :class:`.FunctionSpace` as the first argument to obtain a randomised :class:`.Function`.\n"
                            "\n"
                            "    .. note ::\n"
                            "        FunctionSpace, V, has to be passed as\n"
                            "        the first argument.\n"
                            "\n"
                            "    **Example**::\n"
                            "\n"
                            "        from firedrake import *\n"
                            "        mesh = UnitSquareMesh(2,2)\n"
                            "        V = FunctionSpace(mesh, 'CG', 1)\n"
                            "        pcg = PCG64(seed=123456789)\n"
                            "        rg = Generator(pcg)\n"
                            "        f_beta = rg.beta(V, 1.0, 2.0)\n"
                            "        print(f_beta.dat.data)\n"
                            "        # produces:\n"
                            "        # [0.56462514 0.11585311 0.01247943 0.398984 0.19097059 0.5446709 0.1078666 0.2178807 0.64848515]\n"
                            "\n")

        def __init__(self, bit_generator=None):
            if bit_generator is None:
                from firedrake.randomfunctiongen import PCG64
                bit_generator = PCG64()
            super(_Wrapper, self).__init__(bit_generator)
        _dict['__init__'] = __init__

        # Use decorator to add doc strings to
        # auto generated methods
        def add_doc_string(doc_string):
            def f(func):
                func.__doc__ = _reformat_doc(doc_string)
                return func
            return f

        # To have Sphinx generate docs, make the following methods "static"
        for class_attr, _ in inspect.getmembers(randomgen.Generator):
            if class_attr.startswith('_'):
                continue
            elif class_attr in ['bit_generator', ]:
                continue
            elif class_attr in ['bytes', 'dirichlet', 'integers', 'multinomial', 'multivariate_hypergeometric', 'multivariate_normal', 'shuffle', 'permutation']:
                # These methods are not to be used with V.
                # class_attr is mutable, so we have to wrap func with
                # another function to lock the value of class_attr
                def funcgen(c_a):

                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            raise NotImplementedError("%s.%s does not take FunctionSpace as argument" % (module_attr, c_a))
                        else:
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)

                    return func

                _dict[class_attr] = funcgen(class_attr)
            # Other methods here
            elif class_attr in _known_generator_attributes:
                # Here, too, wrap func with funcgen.
                def funcgen(c_a):
                    @add_doc_string(getattr(_Base, c_a).__doc__)
                    def func(self, *args, **kwargs):
                        if len(args) > 0 and isinstance(args[0], FunctionSpace):
                            # Extract size from V
                            if 'size' in kwargs.keys():
                                raise TypeError("Cannot specify 'size' when generating a random function from 'V'")
                            V = args[0]
                            f = Function(V)
                            args = args[1:]
                            with f.dat.vec_wo as v:
                                kwargs['size'] = (v.local_size,)
                                v.array[:] = getattr(self, c_a)(*args, **kwargs)
                            return f
                        else:
                            # forward to the original implementation
                            return getattr(super(_Wrapper, self), c_a)(*args, **kwargs)
                    return func
                _dict[class_attr] = funcgen(class_attr)
            else:
                warnings.warn("Unknown attribute: Firedrake needs to wrap numpy.random.Generator.%s." % class_attr)
        _Wrapper = type(module_attr, (_Base,), _dict)
        return _Wrapper
    elif module_attr == "RandomGenerator":
        from firedrake.randomfunctiongen import Generator
        return Generator
    elif module_attr in ['MT19937', 'Philox', 'PCG64', 'SFC64']:
        _Base = getattr(randomgen, module_attr)
        __doc__ = _reformat_doc(getattr(randomgen, module_attr).__doc__)

        def __init__(self, *args, **kwargs):
            _kwargs = kwargs.copy()
            self._comm = _kwargs.pop('comm', MPI.COMM_WORLD)
            if self._comm.Get_size() > 1 and module_attr not in ['PCG64', 'Philox']:
                raise TypeError("Use 'PCG64', 'Philox', for parallel RNG")
            self._init(*args, **_kwargs)

        def seed(self, *args, **kwargs):
            raise AttributeError("`seed` method is not available in `numpy.random`; if reseeding, create a new bit generator with the new seed.")

        if module_attr == 'PCG64':
            # Use examples in https://bashtage.github.io/randomgen/parallel.html
            # with appropriate changes.
            def _init(self, *args, **kwargs):
                # numpy.random does not allow users to set `seed` and `inc` separately,
                # while randomgen (v.1.19) does. We follow the randomgen interface here.
                _kwargs = kwargs.copy()
                seed = _kwargs.get("seed")
                inc = _kwargs.pop("inc", None)
                # Use entropy to compute seed.
                rank = self._comm.Get_rank()
                if seed is None:
                    if rank == 0:
                        # generate a 128-bit seed
                        sq = randomgen.SeedSequence(entropy=None, pool_size=4)
                        seed = sq.entropy
                    else:
                        seed = None
                    # All processes have to have the same seed
                    seed = self._comm.bcast(seed, root=0)
                # Use rank to generate multiple streams.
                # If 'inc' is to be passed, it is users' responsibility
                # to provide an appropriate value.
                inc = inc or rank
                # nupy.random.PCG64 does not accept `inc` parameter.
                super(_Wrapper, self).__init__(*args, **_kwargs)
                state = self.state
                state['state'] = {'state': seed, 'inc': inc}
                self.state = state
            # Override __doc__:
            __doc__ = ("\n"
                       "    PCG64(seed=None, inc=None)\n"
                       "\n"
                       "    The original documentations are found at `<https://bashtage.github.io/randomgen/bit_generators/pcg64.html>`__ and at `<https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`__, which are reproduced below with appropriate changes.\n"
                       "\n"
                       "    Container for the PCG-64 pseudo-random number generator."
                       "\n"
                       "    PCG-64 is a 128-bit implementation of O'Neill's permutation congruential\n"
                       "    generator ([1]_, [2]_). PCG-64 has a period of :math:`2^{128}` and supports\n"
                       "    advancing an arbitrary number of steps as well as :math:`2^{127}` streams.\n"
                       "\n"
                       "    ``PCG64`` exposes no user-facing API except ``state``,\n"
                       "    ``cffi`` and ``ctypes``. Designed for use in a ``Generator`` object.\n"
                       "\n"
                       "    **Compatibility Guarantee**\n"
                       "\n"
                       "    ``PCG64`` makes a guarantee that a fixed seed will always produce the same\n"
                       "    results.\n"
                       "\n"
                       "    **Parameters**\n"
                       "\n"
                       "    :kwarg seed: Random seed initializing the pseudo-random number generator.\n"
                       "        Can be an integer in [0, 2**128] or ``None`` (the default).\n"
                       "        If `seed` is ``None``, then ``PCG64`` will use entropy to compute\n"
                       "        a 128-bit seed.\n"
                       "    :kwarg inc: Stream to return.\n"
                       "        Can be an integer in [0, 2**128] or ``None`` (the default).  If `inc` is\n"
                       "        ``None``, then the rank of this MPI process is used.  Can be used with\n"
                       "        the same seed to produce multiple streams using other values of inc.\n"
                       "\n"
                       "    .. note ::\n"
                       "\n"
                       "        Supports the method advance to advance the RNG an arbitrary number of\n"
                       "        steps. The state of the PCG-64 RNG is represented by 2 128-bit unsigned\n"
                       "        integers.\n"
                       "\n"
                       "        See ``PCG32`` for a similar implementation with a smaller period.\n"
                       "\n"
                       "    **Parallel Features**\n"
                       "\n"
                       "    ``PCG64`` can be used in parallel applications in one of two ways.\n"
                       "    The preferable method is to use sub-streams, which are generated by using the\n"
                       "    same value of ``seed`` and incrementing the second value, ``inc``.\n"
                       "\n"
                       "    .. code-block:: python3\n"
                       "\n"
                       "        from firedrake import Generator, PCG64\n"
                       "        rg = [Generator(PCG64(1234, i + 1)) for i in range(10)]\n"
                       "\n"
                       "    The alternative method is to call ``advance`` with a different value on\n"
                       "    each instance to produce non-overlapping sequences.\n"
                       "\n"
                       "    .. code-block:: python3\n"
                       "\n"
                       "        rg = [Generator(PCG64(1234, i + 1)) for i in range(10)]\n"
                       "        for i in range(10):\n"
                       "            rg[i].brng.advance(i * 2**64)\n"
                       "\n"
                       "    **State and Seeding**\n"
                       "\n"
                       "    The ``PCG64`` state vector consists of 2 unsigned 128-bit values,\n"
                       "    which are represented externally as python longs (2.x) or ints (Python 3+).\n"
                       "    ``PCG64`` is seeded using a single 128-bit unsigned integer\n"
                       "    (Python long/int). In addition, a second 128-bit unsigned integer is used\n"
                       "    to set the stream.\n"
                       "\n"
                       "    **References**\n"
                       "\n"
                       '    .. [1] "PCG, A Family of Better Random Number Generators"\n'
                       "           <http://www.pcg-random.org/>_\n"
                       '    .. [2] O\'Neill, Melissa E. "PCG: A Family of Simple Fast Space-Efficient\n'
                       '           Statistically Good Algorithms for Random Number Generation"\n'
                       "           <https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf>_\n"
                       "\n")
        elif module_attr == 'Philox':
            def _init(self, *args, **kwargs):
                seed = kwargs.get("seed")
                # counter = kwargs.get("counter")
                key = kwargs.get("key")
                if self._comm.Get_size() > 1:
                    rank = self._comm.Get_rank()
                    if seed is not None:
                        raise TypeError("'seed' should not be used when using 'Philox' in parallel.  A random 'key' is automatically generated and used unless specified.")
                    # if 'key' is to be passed, it is users' responsibility
                    # to provide an appropriate one
                    if key is None:
                        # Use rank to generate multiple streams
                        key = np.zeros(2, dtype=np.uint64)
                        key[0] = rank
                _kwargs = kwargs.copy()
                _kwargs["key"] = key
                super(_Wrapper, self).__init__(*args, **_kwargs)
        else:
            def _init(self, *args, **kwargs):
                super(_Wrapper, self).__init__(*args, **kwargs)
        _dict = {"__init__": __init__,
                 "_init": _init,
                 "seed": seed,
                 "__doc__": __doc__}
        _Wrapper = type(module_attr, (_Base,), _dict)
        return _Wrapper
    elif module_attr in ['BitGenerator', 'RandomState', 'SeedSequence', 'default_rng', 'get_state', 'seed', 'set_state']:
        return getattr(randomgen, module_attr)
    elif not module_attr.startswith('_'):
        # module_attr not in _known_attributes + _deprecated_attributes
        warnings.warn("Found unknown attribute: Falling back to numpy.random.%s, but Firedrake might need to wrap this attribute." % module_attr)
        return getattr(randomgen, module_attr)


# Module level __getattr__ is only available with 3.7+

import sys

if sys.version_info < (3, 7, 0):
    class Wrapper(object):
        __all__ = __all__

        def __getattr__(self, attr):
            return __getattr__(attr)

    sys.modules[__name__] = Wrapper()
