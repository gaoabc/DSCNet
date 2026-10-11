import torch
import torch.nn.functional as F


def compute_class_centroids(pred, mask, num_classes=2):
    pred = pred.softmax(dim=1)
    B, C, H, W = pred.shape

    centroids = []
    for c in range(num_classes):
        region = pred * (mask == c).float().unsqueeze(1)
        area = region.sum(dim=(2,3), keepdim=True)
        centroid = region.sum(dim=(2,3), keepdim=True) / (area + 1e-6)
        centroids.append(centroid.mean(dim=0))

    return torch.stack(centroids, dim=0)


def confidence_aware_contrastive_loss(features, labels, centroids, confidence, temperature=0.2, conf_threshold=0.5):
    features = features.softmax(dim=1)
    B, C, H, W = features.shape

    valid = (confidence > conf_threshold).float()

    loss = 0

    for c in range(C):
        f = features[:,c]
        pos = centroids[c].squeeze()
        sim = F.cosine_similarity(f, pos.unsqueeze(0), dim=1)

        loss += ((-torch.log(torch.exp(sim/temperature))) * valid).mean()

    return loss
