# csc413-Final-Project

Repository for the final project of CSC413.

# Table of Contents

- [csc413-Final-Project](#csc413-final-project)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Model Figure](#model-figure)
- [Model Parameters](#model-parameters)
- [Model Examples](#model-examples)
- [Data Source](#data-source)
- [Data Summary](#data-summary)
- [Data Transformation](#data-transformation)
- [Data Split](#data-split)
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

# Model Figure

We are using a custom trained transformer that we trained from scratch. 
![Transfromer](./README_FILES/transformer.webp)

# Model Parameters

# Model Examples

# Data Source

https://huggingface.co/datasets/scientific_papers

# Data Summary 

# Data Transformation

# Data Split

# Training Curve

# hyperparameter Tuning

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
