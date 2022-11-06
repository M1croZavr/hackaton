import torch
import torchvision
from torchvision import transforms
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pathology_random_synthesis(x, generator, style_encoder, domains_path, ref_domain='cancer'):
    """Make synthesis with random image from ref_domain as reference image"""
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
    ])
    dataset = torchvision.datasets.ImageFolder(domains_path, transform)
    # reference_distances = torch.tensor([
    #     torch.sum(
    #         (x - x_ref) ** 2
    #     ) if dataset.classes[y_ref] == ref_domain else float('inf')
    #     for x_ref, y_ref in dataset 
    # ])
    reference_indexes = [idx for idx, (x_ref, y_ref) in enumerate(dataset) if dataset.classes[y_ref] == ref_domain]
    # x_ref, y_ref = dataset[torch.argmin(reference_distances)]
    x_ref, y_ref = dataset[random.choice(reference_indexes)]
    x = x.unsqueeze(dim=0).to(DEVICE)
    x_ref, y_ref = x_ref.unsqueeze(dim=0).to(DEVICE), torch.LongTensor([y_ref])
    with torch.inference_mode():
        style_encoder.eval()
        generator.eval()
        style_code = style_encoder(x_ref, y_ref)
        generated_image = generator(x, style_code)
    return generated_image, x_ref
    