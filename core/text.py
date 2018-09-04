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