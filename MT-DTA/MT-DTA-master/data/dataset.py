import torchtext.data as data

class ParallelDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_examples, trg_examples, fields, **kwargs):

        if not isinstance(fields[0], (tuple, list)):
            if trg_examples is None:
                fields = [('src', fields[0])]
            else:
                fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        if trg_examples is None:
            for src_line in src_examples:
                examples.append(data.Example.fromlist(
                    [src_line], fields))
        else:
            for src_line, trg_line in zip(src_examples, trg_examples):
                examples.append(data.Example.fromlist(
                    [src_line, trg_line], fields))

        super(ParallelDataset, self).__init__(examples, fields, **kwargs)