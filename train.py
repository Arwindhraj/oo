import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.encoder import Encoder
from models.decoder import Decoder
from utils.dataset import Flickr30kDataset, get_transform
from utils.utils import collate_fn
from tqdm import tqdm
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():

    root_dir = 'D:/Project_Files/imagecaption/archive/Images'
    captions_file = 'D:/Project_Files/imagecaption/archive/captions.txt'

    embed_size = 512
    vocab_threshold = 5
    vocab_size = 10000
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-4


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform()

    dataset = Flickr30kDataset(root_dir=root_dir, captions_file=captions_file, transform=transform, freq_threshold=vocab_threshold)
    vocab_size = len(dataset.vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    encoder = Encoder(embed_size=embed_size).to(device)
    decoder = Decoder(embed_size=embed_size, vocab_size=vocab_size).to(device)
    print("parameter : 119358264")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate)

    for epoch in range(num_epochs):
        encoder.train() 
        decoder.train()
        epoch_loss = 0
        loop = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, captions in loop:
            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoints/epoch_{epoch+1}.pth')

        if epoch == 0:
            with open('vocab.json', 'w') as vocab_file:
                json.dump({
                    'stoi': dataset.vocab.stoi,
                    'itos': dataset.vocab.itos
                }, vocab_file)

if __name__ == '__main__':
    main()