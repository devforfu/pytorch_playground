import csv
import math
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelBinarizer


_labels_sources = {}


class LabelledImagesDataset:

    def __new__(cls, labels_from: str='file', **kwargs):
        if issubclass(cls, LabelledImagesDataset):
            cls = get_dataset(labels_from)
        return object.__new__(cls)

    def __init__(self, batch_size: int=32,
                 image_extensions: str='*.png|*.jpeg|*.tiff|*.bmp',
                 infinite: bool=True, same_size_batches: bool=False,
                 one_hot: bool=True, reader=None, **kwargs):

        self.batch_size = batch_size
        self.image_extensions = image_extensions
        self.infinite = infinite
        self.same_size_batches = same_size_batches
        self.one_hot = one_hot
        self.reader = reader

        # should be initialized in descendant classes
        self._uid_to_verbose = None
        self._files = None
        self._classes = None
        self._binarizer = None
        self._verbose_classes = None
        self._verbose_to_label = None
        self._label_to_verbose = None
        self._one_hot = None

        self.init()

        self._x_batches = BatchArrayIterator(
            self._files, batch_size=self.batch_size, infinite=self.infinite,
            same_size_batches=self.same_size_batches)

        self._y_batches = BatchArrayIterator(
            self._one_hot, batch_size=self.batch_size, infinite=self.infinite,
            same_size_batches=self.same_size_batches)

    def init(self):
        raise NotImplementedError()

    @property
    def n_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def verbose_classes(self):
        return self._verbose_classes

    def to_label(self, names):
        return np.array([self._verbose_to_label[name] for name in names])

    def to_verbose(self, labels):
        return np.array([self._label_to_verbose[label] for label in labels])

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self._x_batches)
        y = next(self._y_batches)
        if self.reader is not None:
            x = [self.reader(path) for path in x]
        if not self.one_hot:
            y = np.argmax(y, axis=1)
        return x, y


class _LabelsFromFolderNames(LabelledImagesDataset):

    def __init__(self, root: str, **kwargs):
        self.root = root
        super().__init__(**kwargs)

    def init(self):
        uid_to_verbose = {}
        files = []

        for subdir in Path(self.root).iterdir():
            for filename in subdir.iterdir():
                uid = filename.stem
                class_name = subdir.stem
                uid_to_verbose[uid] = class_name
                files.append(filename)

        string_classes = list(uid_to_verbose.values())
        binarizer = LabelBinarizer()
        one_hot = binarizer.fit_transform(string_classes)
        numerical_classes = one_hot.argmax(axis=1)

        self._uid_to_verbose = uid_to_verbose
        self._files = files
        self._classes = np.unique(numerical_classes)
        self._binarizer = binarizer
        self._verbose_classes = np.unique(string_classes)
        self._verbose_to_label = dict(zip(string_classes, numerical_classes))
        self._label_to_verbose = {
            v: k for k, v in self._verbose_to_label.items()}
        self._one_hot = one_hot


def register_source(name, cls):
    global _labels_sources
    _labels_sources[name] = cls


def get_dataset(name):
    if name not in _labels_sources:
        raise ValueError('dispatcher with name \'%s\' is not found' % name)
    return _labels_sources[name]


register_source('folders', _LabelsFromFolderNames)


def read_labels(filename: str, class_column: str, id_column: str='id',
                skip_header: bool=True):
    """Reads CSV file with labels.
    The file should have at least two columns: the one with unique identifiers
    and the another one - with class names.
    Args:
        filename: Path to file with labels.
        class_column: Column with class names.
        id_column: Column with unique identifiers.
        skip_header: If True, then the first row in the file is ignored.
    Returns:
        labels: The mapping from ID to verbose label.
    """
    path = Path(str(filename))

    if not path.exists():
        raise ValueError('labels file is not found: %s' % filename)

    with open(path.as_posix()) as file:
        reader = csv.DictReader(file, fieldnames=[id_column, class_column])
        if skip_header:
            _ = next(reader)
        try:
            labels = {
                strip_exts(row[id_column]): row[class_column]
                for row in reader}
        except KeyError:
            raise ValueError(
                'please check your CSV file to make sure that \'%s\' and '
                '\'%s\' columns exist' % (id_column, class_column))
        else:
            return labels


def strip_exts(filename, exts=None, strip_all=True):
    """Removes specific extensions from file name."""

    if filename.startswith('.'):
        return filename

    if strip_all and exts is None:
        basename, *_ = filename.split('.')
        return basename

    elif not strip_all and exts is None:
        return filename

    list_of_extensions = exts.split('|') if '|' in exts else [exts]
    for ext in list_of_extensions:
        filename = filename.replace('.%s' % ext, '')
    return filename


class FilesIterator:

    def __init__(self, folder: str, pattern: str, batch_size: int=32,
                 infinite: bool=False, same_size_batches: bool=False):

        self.folder = str(folder)
        self.pattern = pattern
        self.infinite = infinite
        self.same_size_batches = same_size_batches
        self.batch_size = batch_size

        extensions = pattern.split('|') if '|' in pattern else [pattern]
        files = list(glob(self.folder, extensions))

        self._extensions = extensions
        self._files = files
        self._n = len(self._files)
        self._iter = BatchArrayIterator(
            self._files, batch_size=batch_size, infinite=infinite)

    @property
    def batch_index(self):
        return self._iter.batch_index

    @property
    def epoch_index(self):
        return self._iter.epoch_index

    @property
    def extensions(self):
        return self._extensions

    def next(self):
        return next(self._iter)


class BatchArrayIterator:
    """Iterates an array or several arrays in smaller batches.

    Attributes:
        batch_size: Size of batch.
        infinite: If True, then the iterator doesn't raise StopIteration
            exception when the array is completely traversed but restarts the
            process again.
        same_size_batches: If True and `infinite` attribute is True, then all
            the batches yielded by the iterator have the same size even if
            the total length of the iterated array is not evenly divided by the
            `batch_size`. If the last batch is smaller then `batch_size`, it is
            discarded.

    """
    def __init__(self,
                 array, *arrays,
                 batch_size: int=32,
                 infinite: bool=False,
                 same_size_batches: bool=False):

        if not infinite and same_size_batches:
            raise ValueError('Incompatible configuration: cannot guarantee '
                             'same size of batches when yielding finite '
                             'number of files.')

        arrays = _convert_to_arrays(array, *arrays)

        self.arrays = arrays
        self.batch_size = batch_size
        self.infinite = infinite
        self.same_size_batches = same_size_batches

        self._n = _num_of_batches(arrays, batch_size, same_size_batches)
        self._batch_index = 0
        self._epoch_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @property
    def n_batches(self):
        return self._n

    @property
    def batch_index(self):
        return self._batch_index

    @property
    def epoch_index(self):
        return self._epoch_index

    def next(self):
        if self._batch_index >= self._n:
            if not self.infinite:
                raise StopIteration()
            self._batch_index = 0
            self._epoch_index += 1

        batches = tuple([self._take_next_batch(arr) for arr in self.arrays])
        self._batch_index += 1
        return batches[0] if len(batches) == 1 else batches

    def _take_next_batch(self, array):
        start = self._batch_index * self.batch_size
        end = (self._batch_index + 1) * self.batch_size
        return array[start:end]


def _convert_to_arrays(seq, *seqs):
    sequences = [seq] + list(seqs)
    arrays = [np.asarray(seq) for seq in sequences]
    n = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n:
            raise ValueError('arrays should have the same length')
    return arrays


def _num_of_batches(arrays, batch_size, same_size):
    n = len(arrays[0])
    if same_size:
        return n // batch_size
    return int(math.ceil(n / batch_size))


def glob(folder, extensions):
    for ext in extensions:
        for path in Path(folder).glob('*.' + ext):
            yield path.as_posix()
