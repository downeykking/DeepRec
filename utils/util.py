import random
import numpy as np


def randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    """Sample random integers from [0, high).
    Args:
        high (int): The largest integer (exclusive) to be drawn from the distribution.
        size (int): The number of samples to be drawn.
        replace (bool): Whether the sample is with or without replacement.
        p: 1-D array-like, optional. The probabilities associated with each entry in [0, high).
           If not given the sample assumes a uniform distribution.
        exclusion: 1-D array-like. The integers in exclusion will be excluded.
    Returns:
        int or ndarray
    """
    if p is None and exclusion is None:
        if size == 1:
            return np.random.randint(0, high=high)
        return np.random.choice(high, size, replace=replace)

    elif exclusion is not None:
        items = np.zeros(size, dtype=np.int32)
        cnt = 0
        while True:
            num = np.random.randint(0, high=high)
            if num not in exclusion:
                items[cnt] = num
                cnt += 1
                if cnt == size:
                    break
        return items[0] if size == 1 else items

# user_idx = randint_choice(100, size=100, replace=True, exclusion=[1, 2, 3])
# print(user_idx)
# from collections.abc import Iterable
# users = user_idx.tolist()


def pad_sequences(sequences, value=-1, max_len=None,
                  padding='post', truncating='post', dtype=int):
    """Pads sequences to the same length.
    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int, float): Padding value. Defaults to `-1`.
        max_len (int or None): Maximum length of all sequences.
        padding (str): `"pre"` or `"post"`: pad either before or after each
            sequence. Defaults to `post`.
        truncating (str): `"pre"` or `"post"`: remove values from sequences
            larger than `max_len`, either at the beginning or at the end of
            the sequences. Defaults to `post`.
        dtype: Type of the output sequences. Defaults to `np.int32`.
    Returns:
        np.ndarray: Numpy array with shape `(len(sequences), max_len)`.
    Raises:
        ValueError: If `padding` or `truncating` is not understood.
    """
    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except:  # noqa
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if max_len is None:
        max_len = np.max(lengths)

    x = np.full([len(sequences), max_len], value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-max_len:]
        elif truncating == 'post':
            trunc = s[:max_len]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# (942, array([  63,   11,  110,  791,   17,  366,  462,   10,   23,  364,  688,
#         293,   33,  932,   39, 1029,  719, 1043,  887,  267,  610,  300,
#         833,  315,  882,  119,  372,  621,    0,   93,  405,    1,  174,
#         161,   57,  135,  576,   48,   73,  217,  213,  138,  221,   84,
#         214,    6,  431,   67,    4,  142,  403,   95,   86,  383,  141,
#         430,  218,   81,  158,  147,  650,   55,  103,  215,  102,  730,
#          83,  530,  229,  528,  522,  143,  941,  386,  648,  133,  740,
#         388,  591,  428,  379,  944,  380,   58,  731,  611,  533,  207,
#         232,  389,  836,  375,  609, 1049,  148,   98,  163,  154,  646,
#         407,  234,  445,  155,  734,  166,  139,  237,  801,  670,  644,
#         653,  866,  157,  656,  156,  815,  385,  745,  226,  892,  672,
#        1180, 1080,  146,  374, 1228,  381,  894,   36, 1325,   89,  184,
#         176,  239,  132,


import inspect
import re


# ref: https://stackoverflow.com/questions/592746/how-can-you-print-a-variable-name-in-python#
def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_results(all_results, metrics, topks, logger):
    for metric in metrics:
        result = {f"{metric:<6}@{topks[i]}": f"{all_results[metric][i]:.4f}" for i in range(len(topks))}
        logger.info(f'(Test Result): {metric.upper():<6}: {result}')
