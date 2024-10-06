# print_captions.py

import os
import torch
from utils.dataset import Flickr30kDataset, get_transform
from torch.utils.data import DataLoader
from utils.utils import collate_fn

def print_captions():
    root_dir = 'D:/Project_Files/imagecaption/archive/Images'
    captions_file = 'D:/Project_Files/imagecaption/archive/captions.txt'
    vocab_threshold = 5

    transform = get_transform()
    dataset = Flickr30kDataset(
        root_dir=root_dir, 
        captions_file=captions_file, 
        transform=transform, 
        freq_threshold=vocab_threshold
    )

    print(f"Total image-caption pairs: {len(dataset)}")
    for idx in range(min(5, len(dataset))):
        image, numericalized_caption, image_id = dataset[idx]  # Unpack three values
        print(f"Caption Length: {len(numericalized_caption)}")
        caption_tokens = [dataset.vocab.itos[token] for token in numericalized_caption.tolist()]
        print(f"Image ID: {image_id}")
        print(f"Caption {idx}: {' '.join(caption_tokens)}\n")

def check_captions():
    root_dir = 'D:/Project_Files/imagecaption/archive/Images'
    captions_file = 'D:/Project_Files/imagecaption/archive/captions.txt'
    vocab_threshold = 5

    dataset = Flickr30kDataset(
        root_dir=root_dir, 
        captions_file=captions_file, 
        freq_threshold=vocab_threshold
    )

    for image_id in dataset.image_ids:
        num_captions = len(dataset.captions[image_id])
        print(f"Image: {image_id}, Number of Captions: {num_captions}")

def print_training_details():
    root_dir = 'D:/Project_Files/imagecaption/archive/Images'
    captions_file = 'D:/Project_Files/imagecaption/archive/captions.txt'
    vocab_threshold = 5
    batch_size = 2
    num_workers = 2
    transform = get_transform()
    dataset = Flickr30kDataset(
        root_dir=root_dir, 
        captions_file=captions_file, 
        transform=transform, 
        freq_threshold=vocab_threshold
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )

    for images, captions, image_ids in dataloader:  # Unpack three values
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"First caption: {captions[0]}")
        print(f"First Image ID: {image_ids[0]}\n")
        break

if __name__ == '__main__':
    print_captions()
    # check_captions()
    # print_training_details()