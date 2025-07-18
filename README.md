# An Empirical Re-evaluation of call graph pruning
This repository contains the scripts used to generate the dataset and machine learning approaches used in the paper

## static_cg_generation
This folder contains the source code of running WALA, DOOP and Opal with various configurations as well as partial orders generation for each tool.
You can find the configurations used to generate the call graphs under \config in each tool with version from 1 to 3.

## dataset_generation
This directory contains the scripts used to generate the dataset such as
- manual_sampling : scripts used to perform stratified sampling
- semantic_features: scripts used to generate semantic features (raw, finetuned)
- structured_features: scripts used to generate structured features

## approach
This directory contains all the experiments in the paper. 

## paper
contains the scripts to generate images in the paper
