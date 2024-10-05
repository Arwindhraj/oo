import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import csv
from .vocabulary import Vocabulary

class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        """
        Args:
            root_dir: Directory with all the images.
            captions_file: Path to the txt file with captions.
            transform: Image transformations.
            freq_threshold: Minimum frequency to include a word in the vocabulary.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Initialize and build the vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.captions = self.load_captions(captions_file)
        self.vocab.build_vocabulary([caption for captions in self.captions.values() for caption in captions])
        self.image_ids = list(self.captions.keys())

    def load_captions(self, captions_file):
        """
        Loads captions from a txt file and organizes them into a dictionary.

        The txt file should have the following format:
            image,caption
            image1.jpg, Caption for image 1.
            image1.jpg, Another caption for image 1.
            image2.jpg, Caption for image 2.
            ...

        Args:
            captions_file: Path to the txt file with captions.

        Returns:
            A dictionary mapping image filenames to a list of their captions.
        """
        captions = {}
        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                if len(row) < 2:
                    continue  # Skip malformed lines
                image_id, caption = row[0].strip(), row[1].strip()
                if image_id in captions:
                    captions[image_id].append(caption)
                else:
                    captions[image_id] = [caption]
        return captions

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_id)
        
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Get one of the captions randomly
        captions_list = self.captions[image_id]
        caption = captions_list[0]  # You can modify this to select a random caption if desired

        # Numericalize the caption
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])