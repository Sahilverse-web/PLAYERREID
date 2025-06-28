import cv2
import torch
import numpy as np
import torchvision.transforms as T

# Dummy ResNet18 embedding backbone (placeholder)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_embedding(image):
    try:
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            return None
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(tensor)
        return embedding.flatten().numpy()
    except Exception as e:
        print("[⚠️] Embedding extraction error:", e)
        return None

def match_embedding(embedding, db, threshold=0.6):
    min_dist = float('inf')
    matched_id = None
    for pid, emb in db.items():
        dist = np.linalg.norm(embedding - emb)
        if dist < threshold and dist < min_dist:
            min_dist = dist
            matched_id = pid
    return matched_id
