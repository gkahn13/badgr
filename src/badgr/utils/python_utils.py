from collections import defaultdict
import copy
import importlib.util
import signal
import sys
import termios
import time
import tty


def import_config(config_fname):
    assert config_fname.endswith('.py')
    spec = importlib.util.spec_from_file_location('config', config_fname)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.params


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def add_recursive(self, keys, value):
        """
        :param keys list(str):
        :param value (anything):
        """
        if isinstance(keys, str):
            keys = keys.split('/')

        assert len(keys) > 0

        d = self
        for key in keys[:-1]:
            if not hasattr(d, key):
                d[key] = AttrDict()
            d = d[key]

        d[keys[-1]] = value

    def get_recursive(self, keys):
        """
        :param keys list(str):
        :return: value
        """
        if isinstance(keys, str):
            keys = keys.split('/')

        d = self
        for key in keys:
            assert key in d
            d = d[key]
        return d

    def ls(self, prefix=''):
        """
        Prints all keys, and recurses into children AttrDicts
        :param prefix (str): printed before each key
        """
        for key in sorted(self.keys()):
            value = self[key]
            if isinstance(value, AttrDict):
                value.ls(prefix=prefix+key+'.')
            else:
                print(prefix + key)

    def apply_recursive(self, func):
        """
        Applies func to each value (recursively) and returns a new AttrDict
        :param func (lambda): takes in one argument and returns one object
        :return AttrDict
        """
        def _apply_recursive(func, d, d_applied):
            for key, value in d.items():
                if isinstance(value, AttrDict):
                    d_applied[key] = value_applied =  AttrDict()
                    _apply_recursive(func, value, value_applied)
                else:
                    d_applied[key] = func(value)

        d_applied = AttrDict()
        _apply_recursive(func, self, d_applied)
        return d_applied

    def modify_recursive(self, func):
        """
        Applies func to each value (recursively), modifying in-place
        :param func (lambda): takes in one argument and returns one object
        """
        for key, value in self.items():
            if isinstance(value, AttrDict):
                value.modify_recursive(func)
            else:
                self[key] = func(value)

    def assert_recursive(self, func):
        """
        Recursively asserts func on each value
        :param func (lambda): takes in one argument, outputs True/False
        """
        for key, value in self.items():
            if isinstance(value, AttrDict):
                value.assert_recursive(func)
            else:
                assert func(value)

    def filter_recursive(self, func):
        d = AttrDict()
        for key, value in self.get_leaf_items():
            if func(key, value):
                d.add_recursive(key, value)
        return d

    def get_leaf_keys(self):
        def _get_leaf_keys(d, prefix=''):
            for key, value in d.items():
                new_prefix = prefix + '/' + key if len(prefix) > 0 else key
                if isinstance(value, AttrDict):
                    yield from _get_leaf_keys(value, prefix=new_prefix)
                else:
                    yield new_prefix

        yield from _get_leaf_keys(self)

    def get_leaf_values(self):
        for key, value in self.items():
            if isinstance(value, AttrDict):
                yield from value.get_leaf_values()
            else:
                yield value

    def get_leaf_items(self):
        for key in self.get_leaf_keys():
            yield key, self.get_recursive(key)

    def __dir__(self):
        return sorted(list(set(super(AttrDict, self).__dir__())) + \
               sorted(list(set(self.keys()))))

    def copy(self):
        d = AttrDict()
        for key, value in self.get_leaf_items():
            d.add_recursive(copy.deepcopy(key), copy.deepcopy(value))
        d.__dict__ = d
        return d

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    @staticmethod
    def from_dict(d):
        d_attr = AttrDict()
        for key, value in d.items():
            d_attr.add_recursive(key.split('/'), value)
        return d_attr

    @staticmethod
    def combine(ds, func):
        leaf_keys = tuple(sorted(ds[0].get_leaf_keys()))
        for d in ds[1:]:
            assert leaf_keys == tuple(sorted(d.get_leaf_keys()))

        d_combined = AttrDict()
        for k in leaf_keys:
            values = [d.get_recursive(k) for d in ds]
            value = func(values)
            d_combined.add_recursive(k, value)

        return d_combined

class TimeIt(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

        self._with_name_stack = []

    def __call__(self, name):
        self._with_name_stack.append(name)
        return self

    def __enter__(self):
        self.start(self._with_name_stack[-1])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        timeit.stop(self._with_name_stack.pop())

    def start(self, name):
        assert(name not in self.start_times)
        self.start_times[name] = time.time()

    def stop(self, name):
        assert(name in self.start_times)
        self.elapsed_times[name] += time.time() - self.start_times[name]
        self.start_times.pop(name)

    def elapsed(self, name):
        return self.elapsed_times[name]

    def reset(self):
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def __str__(self):
        s = ''
        names_elapsed = sorted(self.elapsed_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in names_elapsed:
            if 'total' not in self.elapsed_times:
                s += '{0}: {1: <10} {2:.1f}\n'.format(self.prefix, name, elapsed)
            else:
                assert(self.elapsed_times['total'] >= max(self.elapsed_times.values()))
                pct = 100. * elapsed / self.elapsed_times['total']
                s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, name, elapsed, pct)
        if 'total' in self.elapsed_times:
            times_summed = sum([t for k, t in self.elapsed_times.items() if k != 'total'])
            other_time = self.elapsed_times['total'] - times_summed
            assert(other_time >= 0)
            pct = 100. * other_time / self.elapsed_times['total']
            s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, 'other', other_time, pct)
        return s

timeit = TimeIt()

class Getch:
    @staticmethod
    def getch(block=True):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def exit_on_ctrl_c():
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


class Rate(object):

    def __init__(self, rate):
        self._rate = float(rate)
        self._dt = 1. / self._rate

        self._last_sleep_time = None

    def sleep(self):
        if self._last_sleep_time is not None:
            elapsed_time = time.time() - self._last_sleep_time
            if elapsed_time < self._dt:
                time.sleep(self._dt - elapsed_time)

        self._last_sleep_time = time.time()
