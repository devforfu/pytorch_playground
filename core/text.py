from pathlib import Path

from torchtext.data import Field


class Dataset:

    def __init__(self, field: Field, min_freq: int=1):
        self.field = field
        self.min_freq = min_freq
        self.subsets = {}
        self.vocab_size = None

    def build(self, train: str, valid: str, iterator_factory):
        content_per_file = {}
        for name, path in [('train', train), ('valid', valid)]:
            file_content = []
            new_line = False
            with open(path) as file:
                for line in file:
                    if line.endswith('\n'):
                        new_line = True
                        if line == '\n':
                            continue
                    file_content += self.field.preprocess(line)
                    if new_line:
                        file_content.append(' ')
                        new_line = False
            content_per_file[name] = file_content

        train_text = content_per_file['train']
        self.field.build_vocab(train_text, min_freq=self.min_freq)
        self.vocab_size = len(self.field.vocab.itos)

        for name, content in content_per_file.items():
            sequence = self.field.numericalize(content)
            iterator = iterator_factory(sequence.view(-1))
            self.subsets[name] = iterator

    def __getitem__(self, item):
        if item not in self.subsets:
            raise ValueError(f'Unexpected dataset name: {item}')
        return self.subsets[item]


class TextDataset:

    def __init__(self, field: Field, min_freq: int=1, append_eos: bool=True,
                 keep_new_lines=False, search_pattern='*.txt'):

        self.field = field
        self.min_freq = min_freq
        self.append_eos = append_eos
        self.keep_new_lines = keep_new_lines
        self.search_pattern = search_pattern
        self.vocab_size = None
        self.subsets = {}

    def build(self, train: str, iterator_factory, valid: str=None,
              test: str=None):

        directories = [('train', train), ('valid', valid), ('test', test)]
        content_per_folder = {}

        for name, folder in directories:
            if folder is None:
                continue
            content = []
            for filename in Path(folder).glob(self.search_pattern):
                new_line = False
                with open(filename) as file:
                    for line in file:
                        if line.endswith('\n'):
                            new_line = True
                            if line == '\n':
                                continue
                        content += self.field.preprocess(line)
                        if new_line:
                            char = '\n' if self.keep_new_lines else ' '
                            content.append(char)
                            new_line = False
                if self.append_eos:
                    content.append(['<eos>'])
            content_per_folder[name] = content

        train_text = content_per_folder['train']
        self.field.build_vocab(train_text, min_freq=self.min_freq)
        self.vocab_size = len(self.field.vocab.itos)

        for name, content in content_per_folder.items():
            sequence = self.field.numericalize(content)
            iterator = iterator_factory(sequence.view(-1))
            self.subsets[name] = iterator

    def __getitem__(self, item):
        if item not in self.subsets:
            raise ValueError(f'Unexpected dataset name: {item}')
        return self.subsets[item]
