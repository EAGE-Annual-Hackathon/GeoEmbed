from pathlib import Path
import torch

import clip

from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from geoembed.data import ImageDescriptionDataset

# model loading
device = torch.device('cuda:0')
model, preprocess = clip.load("ViT-B/32", device=device)

dataset_dir_path = Path('./data/dataset')
model.load_state_dict(
    torch.load(Path(dataset_dir_path) / 'model.p')
)
model.eval()

# data
description_file_path = dataset_dir_path / 'descriptions.csv'
dataset = ImageDescriptionDataset(
    preprocess,
    dataset_dir_path,
    description_file_path
)

# one description -> image in "database"
input_description = ['seimic attribute vertical chaotic disturbance realted to gas chimney']
input_description = ['acoustic impedance clinoform']

new_desc = clip.tokenize(input_description).to(device)
for all_descriptions, all_images in DataLoader(dataset, batch_size=len(dataset)):
    break
with torch.no_grad():
    all_descriptions = clip.tokenize(all_descriptions).to(device)
    all_images = all_images.to(device)
    logits_per_image, logits_per_text = model(all_images, new_desc)
    logits_per_text = 100 * logits_per_text.squeeze().softmax(0)
    top10 = logits_per_text.topk(10)

# image generation
fig, (ax1, ax2, ax3) = plt.subplots(3, 3)

fig.tight_layout()
fig.suptitle(input_description)
for ii, ax_ in enumerate(ax1):
    ind, value = top10.indices[ii], top10.values[ii]
    ind = ind.item()
    value = value.item()
    desc_, img_path_ = dataset.get_description_and_image_path(ind)
    ax_.imshow(Image.open(img_path_))
    ax_.set_title(f'Confidence: {value:.2f}%')
    ax_.axis('off')

for ii, ax_ in enumerate(ax2):
    ind, value = top10.indices[ii + 3], top10.values[ii + 3]
    ind = ind.item()
    value = value.item()
    desc_, img_path_ = dataset.get_description_and_image_path(ind)
    ax_.imshow(Image.open(img_path_))
    ax_.set_title(f'Confidence: {value:.2f}')
    ax_.axis('off')

for ii, ax_ in enumerate(ax3):
    ind, value = top10.indices[ii + 6], top10.values[ii + 6]
    ind = ind.item()
    value = value.item()
    desc_, img_path_ = dataset.get_description_and_image_path(ind)
    ax_.imshow(Image.open(img_path_))
    ax_.set_title(f'Confidence: {value:.2f}')
    ax_.axis('off')
plt.show()



img_ = Image.open("./data/new_images/gas_chimney.png")
with torch.no_grad():
    new_img = preprocess(img_)[None].to(device)
    text_embeds, image_embeds = dataset.get_embeddings(model, clip.tokenize, device)
    embed = model.encode_image(new_img)
    prob = 100 * torch.sum(image_embeds * embed, dim=1).softmax(-1)
    top10 = prob.topk(10)


fig, (ax1, ax2, ax3) = plt.subplots(3, 3)
fig.tight_layout()

ax1[0].axis('off')
ax1[2].axis('off')

ax1[1].imshow(img_)
ax1[1].set_title(f'Input')
ax1[1].axis('off')

for ii, ax_ in enumerate(ax2):
    ind, value = top10.indices[ii], top10.values[ii]
    ind = ind.item()
    value = value.item()
    desc_, img_path_ = dataset.get_description_and_image_path(ind)
    ax_.imshow(Image.open(img_path_))
    ax_.set_title(f'Confidence: {value:.2f}%')
    ax_.axis('off')

for ii, ax_ in enumerate(ax3):
    ind, value = top10.indices[ii + 3], top10.values[ii + 3]
    ind = ind.item()
    value = value.item()
    desc_, img_path_ = dataset.get_description_and_image_path(ind)
    ax_.imshow(Image.open(img_path_))
    ax_.set_title(f'Confidence: {value:.2f}%')
    ax_.axis('off')
plt.show()


