import functools
import heapq

import multiprocess as mp
import numpy as np
import tqdm


class ParallelMap:

    def __init__(self, num_workers, init=None, initargs=tuple()):
        if num_workers <= 0:
            if init is not None:
                init(*initargs)
            self.exe = None
        else:
            self.exe = mp.Pool(num_workers, initializer=init, initargs=initargs)
        self.num_workers = max(0, num_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __call__(self, f, inputs, pbar=None, cache=None, asiter=False, **kwargs):
        if not isinstance(inputs, (tuple, list)):
            inputs = list(inputs)

        if cache is not None:
            inputs_og = inputs
            indices = [i for i in range(len(inputs_og)) if inputs_og[i] not in cache]
            inputs = [inputs[i] for i in indices]  # subset to items not in cache

        f = functools.partial(f, **kwargs)
        if not inputs:
            results = []  # can happen if everything is cached
        elif self.exe is None:
            results = map(f, inputs)
        else:
            results = self.exe.imap(f, inputs)
        if pbar is not None:
            results = tqdm.tqdm(results, desc=pbar, total=len(inputs))

        if cache is not None:
            i2out = dict(zip(indices, results))
            results = (cache.setdefault(x, i2out.get(i)) for i, x in enumerate(inputs_og))

        return results if asiter else list(results)

    def shutdown(self):
        if self.exe is not None:
            self.exe.close()
            self.exe.join()


class Heap:

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.heap = []

        self.prio = 0

    def poll(self):
        return [x for _, _, x in self.heap]

    def update(self, item, score):
        x = (score, self.prio, item)
        if len(self.heap) >= self.maxlen:
            heapq.heappushpop(self.heap, x)
        else:
            heapq.heappush(self.heap, x)
        self.prio += 1  # prioritize newer items


class ListOfMetrics:

    def __init__(self):
        self.metrics = dict()

    def reset(self):
        self.metrics = dict()

    def reduce(self, fn, **kwargs):
        return {k: fn(L, **kwargs).item() for k, L in self.metrics.items()}

    def mean(self):
        return self.reduce(np.mean)

    def mean_and_std(self):
        agg = self.mean()
        agg.update(fix_keys(self.reduce(np.std, ddof=1), suf="_std"))
        return agg

    def update(self, input):
        if not self.metrics:
            self.metrics = {k: [] for k in input}
        assert self.metrics.keys() == input.keys()
        for k, v in input.items():
            self.metrics[k].append(v)


def chain(*funcs):

    def fchain():
        for f in funcs:
            f()

    return fchain


def unique(iterable, exclude=None):
    exclude = {None} if (exclude is None) else exclude
    return sorted(set(iterable) - exclude)


def choices(options, n, rng, p=None):
    indices = rng.choice(len(options), size=n, p=p, replace=False).tolist()
    return [options[i] for i in indices]


def choice(options, rng, p=None):
    if len(options) == 1:
        return options[0]
    return choices(options, n=1, rng=rng, p=p)[0]


def randenumerate(sequence, rng):
    indices = rng.permutation(len(sequence))
    for i in indices:
        yield i, sequence[i]


def normalize(input):
    input = np.asarray(input, dtype=np.float32)
    if input.size == 0:
        return input
    return input / input.sum()


def rank_by_value(iterable, f, rng):
    noise = rng.random(len(iterable))  # random tie-break
    graph = [(f(x), eps, x) for x, eps in zip(iterable, noise)]
    graph.sort(reverse=True)
    return [x for _, _, x in graph]


def fix_keys(D, pre="", suf=""):
    return {f"{pre}{k}{suf}": v for k, v in D.items()}
