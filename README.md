# *ACL Paper Styles

This directory contains the latest LaTeX and Word templates for *ACL
conferences.

## Instructions for authors

Paper submissions to *ACL conferences must use the official ACL style
templates.

The LaTeX style files are available

- as an [Overleaf template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)
- in this repository, in the [`latex`](https://github.com/acl-org/acl-style-files/blob/master/latex) subdirectory
- as a [.zip file](https://github.com/acl-org/acl-style-files/archive/refs/heads/master.zip)

Please see [`latex/acl_latex.tex`](https://github.com/acl-org/acl-style-files/blob/master/latex/acl_latex.tex) for an example.

The Microsoft Word template is available in this repository at [`word/acl.docx`](https://github.com/acl-org/acl-style-files/blob/master/word/acl.docx).

Please follow the paper formatting guidelines general to *ACL
conferences:

- [Paper formatting guidelines](https://acl-org.github.io/ACLPUB/formatting.html)

Authors may not modify these style files or use templates designed for
other conferences.

## Instructions for publications chairs

To adapt the style files for your conference, please fork this repository and
make necessary changes. Minimally, you'll need to update the name of
the conference and rename the files.

If you make improvements to the templates that should be propagated to
future conferences, please submit a pull request. Thank you in
advance!

In older versions of the templates, authors were asked to fill in the
START submission ID so that it would be stamped at the top of each
page of the anonymized version. This is no longer needed, because it
is now possible to do this stamping automatically within
START. Currently, the way to do this is for the program chair to email
support@softconf.com and request it.

## Instructions for making changes to style files

- merge pull request in github, or push to github
- git pull from github to a local repository
- then, git push from your local repository to overleaf project 
    - Overleaf project is https://www.overleaf.com/project/5f64f1fb97c4c50001b60549
    - Overleaf git url is https://git.overleaf.com/5f64f1fb97c4c50001b60549
- then, click "Submit" and then "Sumbit as Template" in overleaf in order to ask overleaf to update the overleaf template from the overleaf project 

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