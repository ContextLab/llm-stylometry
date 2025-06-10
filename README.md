# Repository overview

The repository is organized as follows:

```
.
├── code : all python scripts used in the paper
├── data : all data analyzed in the paper
│   ├── raw : raw data
│   ├── clean : cleaned data
│   ├── book_and_chapter_titles.txt : Oz book and chapter titles (used in cleaning)
└── models : model loss logs and config info
└── figures : figures used in the paper
```

# How to replicate our results

## Setting up the environment

Install `conda`. Then, run:

```
conda create -n llm-stylometry python=3.10
conda activate llm-stylometry
conda install -c pytorch -c nvidia pytorch=2.2.2 pytorch-cuda=12.1 torchtriton=2.2.0
conda install "numpy<2" scipy transformers
```

## Training models
Run:
```
python code/main.py
```

## Creating figures
Run:
```
conda install matplotlib seaborn
python code/all_losses.py
python code/confusion_matrix.py
python code/loss_heatmaps.py
python code/oz_losses.py
python code/stripplot.py
python code/t_test_figs.py
```
The above scripts will save the figures to `figures/`.
