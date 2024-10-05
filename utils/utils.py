import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)

    captions = [item[1] for item in batch]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions