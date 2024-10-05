import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_size, num_classes=20):
        super(Encoder, self).__init__()
        # Vision Transformer for feature extraction
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove the classification head

        # Pretrained Object Detection model
        self.object_detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.object_detector.eval()  # Set to evaluation mode

        # Freeze Object Detector parameters
        for param in self.object_detector.parameters():
            param.requires_grad = False

        # Determine hidden dimension based on torchvision version
        try:
            hidden_dim = self.vit.config.hidden_size  # For HuggingFace-like models (if applicable)
        except AttributeError:
            # Depending on torchvision version, use the appropriate attribute
            hidden_dim = getattr(self.vit, 'hidden_dim', None)
            if hidden_dim is None:
                hidden_dim = getattr(self.vit, 'embed_dim', None)
            if hidden_dim is None:
                raise AttributeError("Cannot determine 'hidden_dim' or 'embed_dim' from VisionTransformer.")

        # Reduce the dimension of features
        self.fc = nn.Linear(hidden_dim + num_classes, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        # Extract image features using Vision Transformer
        vit_features = self.vit(images)  # (batch_size, hidden_size)

        # Ensure object_detector remains in eval mode
        self.object_detector.eval()

        # Extract object labels using Object Detection without computing gradients
        with torch.no_grad():
            detections = self.object_detector(images)

        # Process detections to create a label tensor
        labels = self.process_detections(detections, images.device)  # (batch_size, num_classes)

        # Concatenate features and labels
        combined = torch.cat((vit_features, labels), dim=1)  # (batch_size, hidden_size + num_classes)

        # Pass through fully connected layer
        features = self.fc(combined)  # (batch_size, embed_size)
        features = self.relu(features)
        return features

    def process_detections(self, detections, device):
        # Example processing. Adjust num_classes as needed.
        num_classes = 20  # Define the number of classes you want to consider
        labels_tensor = torch.zeros((len(detections), num_classes), device=device)
        for i, det in enumerate(detections):
            for label in det['labels']:
                if label < num_classes:
                    labels_tensor[i, label] = 1
        return labels_tensor