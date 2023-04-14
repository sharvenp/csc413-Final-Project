# CSC413 Final Project

Repository for the final project of CSC413.

### Table of Contents

- [Introduction](#introduction)
- [Model](#model)
- [Dataset](#data)
- [Training Curve](#training-curve)
- [hyperparameter Tuning](#hyperparameter-tuning)
- [Quantitative Measures](#quantitative-measures)
- [Quantitative and Qualitative Results](#quantitative-and-qualitative-results)
- [Justification of Results](#justification-of-results)
- [Ethical Consideration](#ethical-consideration)
- [Authors](#authors)
  

# Introduction 
We will create a model that is capable of summarizing text. The summarization will be abstractive instead of extractive, meaning that the model will try to generate a paraphrasing of the input data instead of highlighting key points.

We are planning on making our proof of concept by creating a model that, given an abstract of a scientific paper, will predict the title of the paper. Then we will show we can extend it to work on entire papers to create an abstract given more time and computing resources.

# Model

### Model Figure

We are using a custom trained transformer that we trained from scratch. 
![Transfromer](./README_FILES/transformer.webp)

TODO: add actual figure

### Model Parameters

### Model Examples

Examples from the test set:

Successful:
- Input:
- Expected Output:
- Actual Output:

Unsuccessful:
- Input:
- Expected Output:
- Actual Output:

# Data

### Data Source
TODO: Thomas

https://huggingface.co/datasets/scientific_papers

### Data Summary 
TODO: Thomas
Vocab size: 30,000

### Data Transformation
TODO: Thomas

### Data Split

After data transformation, we had a dataset with 1,990,535 examples.

Due to memory contraints as well as training time constraints, we decided to limit the training dataset size to 800,000 examples (40%) of dataset, used around 5000 examples for the validation dataset (0.2%) and around 20,000 examples for the test dataset (10%). We used a small validation set to be able to quickly calculate the validation loss during the training time and use it as an indicator of overfitting. We also used a resonably sized test dataset so we can mostly accurately assess the model's performance on new examples. The exact counts are specified below:

- Train dataset size: 800000
- Val dataset size: 4976
- Test dataset size: 199053

# Training Curve

# Hyperparameter Tuning

Due to memory as well as training time constraints, we were not able to multiple models on the original dataset to tune hyperparameters. However when overfitting the model to a small dataset of 5000 examples, we were able to easily elimate some hyperparameter choices because they either performed poorly, or took too long to converge to a low loss. Hyperparameters we needed to consider:

- Training Batch Size
- Learning Rate 
- Weight Decay (ADAM L2 Penalty)
- Gradient Clipping Maximum (Maximum maginitude of gradient before it is clipped)
- Embedding Dimenson (Dimension for embedding for both encoder and decoder vocabluary embeddings)
- Transformer Hidden Dimension (Number of units in feedforward network model for both encoder and decoder layers)
- Transformer Attention Heads (Number of multihead attention models)
- Transformer Encoder/Decoder Layers (Number of Sub-encoder/Sub-decoder layers in Encoder/Decoder)
- Dropout (Dropout rate for Positional Encoder Output)

Options we experimented with:

Option 1: (Did not overfit to dataset and failed to converge)
- Training Batch Size: 32
- Learning Rate: 1e-3
- Weight Decay: 0
- Gradient Clipping Maximum: 10
- Embedding Dimenson: 128
- Transformer Hidden Dimension: 128
- Transformer Attention Heads: 4
- Transformer Encoder/Decoder Layers: 2
- Dropout: 0.3

Option 2: (Converged too slowly and did not generalize well)
- Training Batch Size: 32
- Learning Rate: 1e-6
- Weight Decay: 1e-6
- Gradient Clipping Maximum: 2
- Embedding Dimenson: 256
- Transformer Hidden Dimension: 512
- Transformer Attention Heads: 4
- Transformer Encoder/Decoder Layers: 2
- Dropout: 0.3

Option 3: (Overgeneralized and was unable to learn)
- Training Batch Size: 32
- Learning Rate: 1e-4
- Weight Decay: 1e-5
- Gradient Clipping Maximum: 2
- Embedding Dimenson: 1024
- Transformer Hidden Dimension: 2048
- Transformer Attention Heads: 4
- Transformer Encoder/Decoder Layers: 4
- Dropout: 0.3

Option 4: (Performed the best)
- Training Batch Size: 64
- Learning Rate: 1e-4
- Weight Decay: 1e-5
- Gradient Clipping Maximum: 5
- Embedding Dimenson: 512
- Transformer Hidden Dimension: 1024
- Transformer Attention Heads: 8
- Transformer Encoder/Decoder Layers: 4
- Dropout: 0.3

*Note that a batch size of 64 was the highest we were able to go before running out of GPU VRAM when training on the actual data*

# Quantitative Measures

# Quantitative and Qualitative Results

# Justification of Results

# Ethical Consideration

We identify multiple groups that could use this model, those with low literacy needing assistance (i.e. disabled people, dyslexic people, children, English as a second language), and researchers trying to find which papers would be best for their work cited. 

For people with reading disabilities, summarizing paragraphs into one sentence can help them get a better understanding of the text, For researchers getting summaries can help them decide if the paper is worth reading or citing for their own work. 

However models can have bias or unintentionally change the meaning of the original text. If this happens for people learning english or otherwise unable to read the original text it can damage their learning or give them false ideas. For researchers it could cause them to skip a paper that would have been useful to them.

# Authors 

We plan to cover most of the model building in a group coding session. This will take place in our discord server where one person will share their screen for Google Collab and we will load the data and start building the model.

We will start by getting the data onto Google Collab, then creating several models with different architectures that we can compare for hyperparameter tuning. Then we will overfit on the first 20 or so papers. This will all be done over one or two days. 

After the initial setup we will follow the following setup: 

Ajitesh: Hyperparameter Tuning, Quantitative Measures, Ethical Consideration 

Sharven: Introduction, Model Figure, Model Parameters, Model Examples

Thomas: Training the model, Data Source, Data Summary, Data Transformation, Data Split

Then finally we will group together to write up on results and publish the final README to the repository.
