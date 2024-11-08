import multiprocessing as mp

class SharedState(object):
    def __init__(self):
        self._mp_manager = mp.Manager()
        self._dict = self._mp_manager.dict()

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def clear(self):
        self._dict.clear()

    def get_dict(self):
        return self._dict

    def __repr__(self):
        return repr(self._dict)

    def __str__(self):
        return str(self._dict)
