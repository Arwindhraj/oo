import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, embed_size, num_classes=20):
        super(Encoder, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  

        self.object_detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.object_detector.eval()  
        
        for param in self.object_detector.parameters():
            param.requires_grad = False
        
        try:
            hidden_dim = self.vit.config.hidden_size  
        except AttributeError:
            
            hidden_dim = getattr(self.vit, 'hidden_dim', None)
            if hidden_dim is None:
                hidden_dim = getattr(self.vit, 'embed_dim', None)
            if hidden_dim is None:
                raise AttributeError("Cannot determine 'hidden_dim' or 'embed_dim' from VisionTransformer.")
        
        self.fc = nn.Linear(hidden_dim + num_classes, embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        
        vit_features = self.vit(images)  
        
        self.object_detector.eval()
        
        with torch.no_grad():
            detections = self.object_detector(images)
        
        labels = self.process_detections(detections, images.device)  
        
        combined = torch.cat((vit_features, labels), dim=1)  
        
        features = self.fc(combined)  
        features = self.relu(features)
        return features

    def process_detections(self, detections, device):
        
        num_classes = 20  
        labels_tensor = torch.zeros((len(detections), num_classes), device=device)
        for i, det in enumerate(detections):
            for label in det['labels']:
                if label < num_classes:
                    labels_tensor[i, label] = 1
        return labels_tensor