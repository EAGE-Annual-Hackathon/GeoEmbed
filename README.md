# GeoEmbed


## Description
Build embedding for geoscience.  
We leverage the power of the CLIP model from OpenAI to create new way of indexing geoscience material.
We hope this will help industry to move toward energy-efficient way of condicting project by enabling the reuse all existing geoscience material.


## Installation

Based on OpenAI CLIP project https://github.com/openai/CLIP 
```
$ pip install Pillow pandas numpy ftfy regex tqdm streamlit
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install git+https://github.com/openai/CLIP.git
$ pip install .

```

## Usage
Directly run scripts
``` 
$ python finetuning.py
$ python inference_and_image_generation.py
```



