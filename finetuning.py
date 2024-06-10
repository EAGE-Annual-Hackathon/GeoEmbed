from pathlib import Path
from tqdm import trange
import torch
import clip

from torch.utils.data import DataLoader
from geoembed.data import ImageDescriptionDataset


dataset_dir_path = Path('./data/dataset')
description_file_path = dataset_dir_path / 'descriptions.csv'
n_epochs = 30
batch_size = 15

# data loading
device = torch.device('cuda:0')
model, preprocess = clip.load("ViT-B/32", device=device)
dataset = ImageDescriptionDataset(
    preprocess,
    dataset_dir_path,
    description_file_path
)
data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
# losses and optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-5,
    betas=(0.9,0.98),
    eps=1e-6,
    weight_decay=0.2
)
loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()
loss_monitor = []

model.train()
for _ in trange(n_epochs):
    for descriptions, images in data_loader:
        descriptions = clip.tokenize(descriptions).to(device)
        images = images.to(device)

        logits_per_image, logits_per_text = model(images, descriptions)
        ground_truth = torch.arange(len(images), device=device)
        total_loss = (
            loss_img(logits_per_image, ground_truth)
            +
            loss_txt(logits_per_text, ground_truth)
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_monitor.append(total_loss.item())

# save model weights
torch.save(
    model.state_dict(),
    dataset_dir_path / 'model.p'
)

