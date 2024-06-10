from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ImageDescriptionDataset(Dataset):

    prefixes = {
        'seis': 'seimic attribute ',
        'phi': 'porosity attribute ',
        'AI': 'acoustic impedance attribute '
    }

    def __init__(
            self,
            preprocess,
            image_dir_path,
            description_file_path,
            add_prefix_information=True
        ) -> None:
        super().__init__()
        self.preprocess = preprocess
        self.df_descriptions = pd.read_csv(description_file_path, index_col='index')
        self.df_descriptions.columns = ['description']
        self.image_paths = list(Path(image_dir_path).rglob('*.png'))
        self.add_prefix_information = add_prefix_information

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        desc, image_path = self.get_description_and_image_path(index)
        return desc, self.preprocess(Image.open(image_path))

    def get_description_and_image_path(self, index):
        image_path = self.image_paths[index]
        desc_index = int(image_path.parent.name)
        desc = self.df_descriptions.loc[desc_index]['description']
        if self.add_prefix_information:
            prefix = image_path.name.split('_')[0]
            desc = self.prefixes[prefix] + desc
        return desc, image_path

    def get_embeddings(self, model, tokenizer, device):
        text_embeds = []
        image_embeds = []
        for index in range(len(self)):
            desc, image_preprocessed = self[index]
            text_embeds.append(
                tokenizer(desc)
            )
            image_embeds.append(
                image_preprocessed
            )
        with torch.no_grad():
            text_embeds = model.encode_text(
                torch.concatenate(text_embeds, dim=0).to(device)
            )
            image_embeds = model.encode_image(
                torch.stack(image_embeds).to(device)
            )
        return text_embeds, image_embeds
