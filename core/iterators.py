import numpy as np


class SequenceIterator:
    """
    A simple iterator to convert an array of encoded characters into group of
    batches reshaped into format, appropriate for the RNN training process.
    """
    def __init__(self, seq, bptt=10, batch_size=64, random_length=True,
                 flatten_target=True):

        # Converting dataset into batches:
        # 1) truncate text length to evenly fit into number of batches
        # 2) reshape the text into N (# of batches) * M (batch size)
        # 3) transpose to convert into "long" format with fixed number of cols

        n_batches = seq.size(0) // batch_size
        truncated = seq[:n_batches * batch_size]
        batches = truncated.view(batch_size, -1).t().contiguous()

        self.bptt = bptt
        self.batch_size = batch_size
        self.random_length = random_length
        self.flatten_target = flatten_target
        self.batches = batches
        self.curr_line = 0
        self.curr_iter = 0
        self.total_lines = batches.size(0)
        self.total_iters = self.total_lines // self.bptt - 1

    @property
    def completed(self):
        if self.curr_line >= self.total_lines - 1:
            return True
        if self.curr_iter >= self.total_iters:
            return True
        return False

    def __iter__(self):
        self.curr_line = self.curr_iter = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.completed:
            raise StopIteration()
        seq_len = self.get_sequence_length()
        batch = self.get_batch(seq_len)
        self.curr_line += seq_len
        self.curr_iter += 1
        return batch

    def get_sequence_length(self):
        """
        Returns a length of sequence taken from the dataset to form a batch.

        By default, this value is based on the value of bptt parameter but
        randomized during training process to pick sequences of characters with
        a bit different length.
        """
        if self.random_length is None:
            return self.bptt
        bptt = self.bptt
        if np.random.random() >= 0.95:
            bptt /= 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        return seq_len

    def get_batch(self, seq_len):
        """
        Picks training and target batches from the source depending on current
        iteration number.
        """
        i, source = self.curr_line, self.batches
        seq_len = min(seq_len, self.total_lines - 1 - i)
        X = source[i:i + seq_len].contiguous()
        y = source[(i + 1):(i + 1) + seq_len].contiguous()
        if self.flatten_target:
            y = y.view(-1)
        return X, y