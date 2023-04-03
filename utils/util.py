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
        items = np.zeros(size)
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


def pad_sequences(sequences, value=0, max_len=None,
                  padding='post', truncating='post', dtype=int):
    """Pads sequences to the same length.
    Args:
        sequences (list): A list of lists, where each element is a sequence.
        value (int, float): Padding value. Defaults to `0.`.
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
