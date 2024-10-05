import torch
from PIL import Image
from torchvision import transforms
from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary
import json

def load_model(encoder_path, decoder_path, vocab_path, device):
    # Initialize models
    encoder = Encoder(embed_size=512).to(device)
    decoder = Decoder(embed_size=512, vocab_size=10000).to(device)  # Adjust vocab_size accordingly

    # Load the saved model parameters
    encoder.load_state_dict(torch.load(encoder_path, map_location=device)['encoder_state_dict'])
    decoder.load_state_dict(torch.load(decoder_path, map_location=device)['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    vocab = Vocabulary(freq_threshold=5)
    vocab.itos = {int(k): v for k, v in vocab.items()["itos"].items()}
    vocab.stoi = {v: int(k) for k, v in vocab.itos.items()}

    return encoder, decoder, vocab

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return image

def generate_caption(encoder, decoder, vocab, image, device, max_length=20):
    with torch.no_grad():
        features = encoder(image.to(device))
        generated = [vocab.stoi["<SOS>"]]

        for _ in range(max_length):
            tgt = torch.tensor(generated).unsqueeze(0).to(device)  # (1, len)
            output = decoder(features, tgt)
            output = output.argmax(2)
            next_word = output[0, -1].item()
            if next_word == vocab.stoi["<EOS>"]:
                break
            generated.append(next_word)

        caption = [vocab.itos[idx] for idx in generated[1:]]
        return ' '.join(caption)

def main():
    # Paths
    encoder_path = 'checkpoints/epoch_10.pth'
    decoder_path = 'checkpoints/epoch_10.pth'
    vocab_path = 'data/Flickr30k/vocab.json'  # Save your vocab as JSON during training
    image_path = 'path_to_your_image.jpg'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models and vocabulary
    encoder, decoder, vocab = load_model(encoder_path, decoder_path, vocab_path, device)

    # Preprocess image
    image = preprocess_image(image_path)

    # Generate caption
    caption = generate_caption(encoder, decoder, vocab, image, device)
    print(f'Caption: {caption}')

if __name__ == '__main__':
    main()

"""
If the above did not work try this:
import torch
from PIL import Image
from torchvision import transforms
from models.encoder import Encoder
from models.decoder import Decoder
from utils.vocabulary import Vocabulary
import json
import argparse

def load_model(encoder_path, decoder_path, vocab_path, device):
    # Initialize models
    encoder = Encoder(embed_size=512).to(device)
    decoder = Decoder(embed_size=512, vocab_size=10000).to(device)  # Adjust vocab_size accordingly

    # Load the saved model parameters
    checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = Vocabulary(freq_threshold=5)
    vocab.stoi = vocab_data['stoi']
    vocab.itos = {int(k): v for k, v in vocab_data['itos'].items()}

    return encoder, decoder, vocab

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return image

def generate_caption(encoder, decoder, vocab, image, device, max_length=20):
    with torch.no_grad():
        features = encoder(image.to(device))
        generated = [vocab.stoi["<SOS>"]]

        for _ in range(max_length):
            tgt = torch.tensor(generated).unsqueeze(0).to(device)  # (1, len)
            output = decoder(features, tgt)
            output = output.argmax(2)
            next_word = output[0, -1].item()
            if next_word == vocab.stoi["<EOS>"]:
                break
            generated.append(next_word)

        caption = [vocab.itos[idx] for idx in generated[1:]]
        return ' '.join(caption)

def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning Inference')
    parser.add_argument('--encoder_path', type=str, default='checkpoints/epoch_10.pth', help='Path to the encoder checkpoint')
    parser.add_argument('--decoder_path', type=str, default='checkpoints/epoch_10.pth', help='Path to the decoder checkpoint')
    parser.add_argument('--vocab_path', type=str, default='data/Flickr30k/vocab.json', help='Path to the vocabulary JSON file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to caption')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models and vocabulary
    encoder, decoder, vocab = load_model(args.encoder_path, args.decoder_path, args.vocab_path, device)

    # Preprocess image
    image = preprocess_image(args.image_path)

    # Generate caption
    caption = generate_caption(encoder, decoder, vocab, image, device)
    print(f'Caption: {caption}')

if __name__ == '__main__':
    main()
"""