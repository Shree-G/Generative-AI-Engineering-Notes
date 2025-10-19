# Tokenization

#### Tokenization Overview

- Tokenization is the process of dividing text into smaller pieces, or tokens, to help AI models understand the text better.
- Different types of tokens can be generated, such as words, characters, or subwords, depending on the tokenization method used.

#### Tokenization Methods

- Word-based tokenization splits text into individual words, preserving semantic meaning but increasing vocabulary size.
- Character-based tokenization divides text into individual characters, resulting in smaller vocabularies but potentially losing meaning.
- Subword-based tokenization keeps frequently used words intact while breaking down infrequent words into meaningful subwords, combining the advantages of the other methods.
	- seems to be the industry standard
	- wordpiece
		- evaluates the benefits/drawbacks of splitting and merging two symbols
		- has ## in front of tokens to indicate that the previous token came before it
	- unigram
		- breaks text into smaller pieces
		- narrows down large list of possibilities by freq of appearance
		- has _ in front of the token if it is the start of a new word
	- sentencepiece
		- segments text into manageable parts and assigns unique IDs

#### Implementation in Libraries

- Tokenizers like NLTK and spaCy are used to generate tokens, while libraries like PyTorch can tokenize text and create a vocabulary for models.
- How to do tokenization with PyTorch:
	-  Go through and split up a sentence into individual words/subwords/letters
	- Creating a mapping for each of these tokens to a numerical value
	- Use this mapping to then output a list of indices.
- Special tokens can be added to tokenized sentences to ensure consistency in processing, such as beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.

# Implementing Tokenization Lab

Technologies Used:
- [`nltk`](https://www.nltk.org/) or natural language toolkit, will be employed for data management tasks. It offers comprehensive tools and resources for processing natural language text, making it a valuable choice for tasks such as text preprocessing and analysis.
    
- [`spaCy`](https://spacy.io/) is an open-source software library for advanced natural language processing in Python. spaCy is renowned for its speed and accuracy in processing large volumes of text data.
    
- [`BertTokenizer`](https://huggingface.co/docs/transformers/main_classes/tokenizer#berttokenizer) is part of the Hugging Face Transformers library, a popular library for working with state-of-the-art pre-trained language models. BertTokenizer is specifically designed for tokenizing text according to the BERT model's specifications.
	- these do not work with torchtext - they are specifically meant for feeding stuff into BERT
	- output of BertTokenizer is a PyTorch tensor that gets fed into a PyTorch model
		- BUT tokenization and data handling are managed within the HuggingFace ecosystem (transformers and datasets)
    
- [`XLNetTokenizer`](https://huggingface.co/docs/transformers/main_classes/tokenizer#xlnettokenizer) is another component of the Hugging Face Transformers library. It is tailored for tokenizing text in alignment with the XLNet model's requirements.
	- these do not work with torchtext - they are specifically meant for feeding stuff into XLNet
    
- [`torchtext`](https://pytorch.org/text/stable/index.html) It is part of the PyTorch ecosystem, to handle various natural language processing tasks. It simplifies the process of working with text data and provides functionalities for data preprocessing, tokenization, vocabulary management, and batching.

### Word based tokenizer
- General libraries like nltk and spaCy often split contractions into individual words
- No universal rule for this, but general guideline is ot preserve the input format after tokenization to match how the model was trained
- Some word based tokenizers are also able to get the meta characteristics of the word in the sentence like the following:
	- I PRON nsubj: "I" is a pronoun (PRON) and is the nominal subject (nsubj) of the sentence.
	- help VERB ROOT: "help" is a verb (VERB) and is the root action (ROOT) of the sentence.
	- afraid ADJ acomp: "afraid" is an adjective (ADJ) and is an adjectival complement (acomp) which gives more information about a state or quality related to the verb.
- The central problem is that words with similar meanings will be assigned to different IDs, resulting in them being treated as spearate words with distinct meanings. Unicorns and unicorn would be tokenized as two separate words

### Subword-based tokenizer
- allows frequently used words to be unsplit, while breaking down infrequent words
- Techniques like SentencePiece or WordPiece learn subword units from a given text corpus
	- identifies common prefixes, suffixes and root words

#### WordPiece
- Initially, WordPiece initializes it's vocabulary to include every character in the training data, and then progressively combines characters together to form it's vocabulary
	- The way it does this is by measuring the "loss" it would incur by merging two symbols/tokens together based on how frequent this combined token appears in the training data

Breakdown of output:
- 'ibm': "IBM" is tokenized as 'ibm'. BERT converts tokens into lowercase, as it does not retain the case information when using the "bert-base-uncased" model.
- 'taught', 'me', '.': These tokens are the same as the original words or punctuation, just lowercased (except punctuation).
- 'token', '##ization': "Tokenization" is broken into two tokens. "Token" is a whole word, and "##ization" is a part of the original word. The "##" indicates that "ization" should be connected back to "token" when detokenizing (transforming tokens back to words).

#### Unigram and SentencePiece
- Unigram
	- method for breaking words/text down into smaller pieces
	- starting with a large list of possibilities, narrowing down based on how frequently those pieces appear in the text
	- efficient text tokenization
- SentencePiece
	- takes test, divides it into smaller, manageable parts and assigns ID's to those segments
	- main thing is it does so consistently
	- no matter how many times you run SentencePiece on the same text, you will obtain the same subwords and IDs
- Both of these work together
	- SentencePiece handles subword segmentation and ID assignment
	- Unigram principles guide the vocab reduction process to create a more efficient representation of the text data

Breakdown of output:
- '▁IBM': The "▁" (often referred to as "whitespace character") before "IBM" indicates that this token is preceded by a space in the original text. "IBM" is kept as is because it's recognized as a whole token by XLNet and it preserves the casing because you are using the "xlnet-base-cased" model.
- '▁taught', '▁me', '▁token': Similarly, these tokens are prefixed with "▁" to indicate they are new words preceded by a space in the original text, preserving the word as a whole and maintaining the original casing.
- 'ization': Unlike "BertTokenizer," "XLNetTokenizer" does not use "##" to indicate subword tokens. "ization" appears as its own token without a prefix because it directly follows the preceding word "token" without a space in the original text.
- '.': The period is tokenized as a separate token since punctuation is treated separately.

### Tokenization with PyTorch
- torchtext from pytorch is a processing pipeline that can incorporate various tokenizers
- its meant to streamline the process of preparing text data for trianing models in PyTorch
	- creates datasets, builds vocabularies, and creates iterators that batch and pad sequences
	- provides basic, easy-to-use tokenizers
- torchtext also integrates with spaCy and NLTK
	- you define a tokenizer function using spaCy or NLTK's models and then pass that function to the torchtext data processing pipeline
	- you do this to use sophisticated, language specific tokenization thta can handle punctuation, hyphens and special cases far better than a simple space-based split
	- spaCy is often said to be best bet for multi-language applications
- after tokenization, the vocab maps these tokens to unique integers - allowing them to be fed into neural networks

Token Indices:
- represent words as numbers because NLP algorithms can process and manipulate numbers more efficiently and quickly than raw text
- use the function build_vocab_from_iterator
	- assigns a unique index to each token based on its position in the vocabulary
-  build a vocab from the tokenized texts generated by the yielf_tokens generator function lazily using an iterator

Out-of-vocabulary
- sometimes there are words not present in the vocabulary because they are rare or unseen during the vocabulary building process
- when encountering these words, the model can use the unk token to represent them
	- by including the unk token, you provide a way to handle out-of-vocabulary words in your language model

# Overview of Data Loaders

Data Loaders Overview

- A data loader is essential for efficiently preparing and loading large datasets, especially in applications like machine translation.
- It facilitates batching and shuffling of data, which is crucial for training neural networks.
- Seamless integration with PyTorch training pipeline

Creating Custom Datasets

- Custom datasets can be created by inheriting from the `torch.utils.data.dataset` class, which includes methods for initialization, length, and item retrieval.
- Data loaders can output data in batches, allowing for more efficient training processes.

Data Transformation and Pre-processing

- Data loaders support on-the-fly pre-processing, including tokenization, numericalization, and padding to ensure uniform input sizes.
- The collate function can be used for custom transformations, ensuring that data is prepared in a suitable format for deep learning models.

# Data Quality and Diversity for LLM Training

### Data Quality
- refers to the accuracy, consistency and completeness of the dataset used for training
- poor quality data might introduce noise

Practices to ensure high quality data:
- Noise Reduction
	- remove irrelevant or repetitive data to help model focus on significant patterns
- Consistency checks
	- regularly verify consistency to prevent conflicting or outdated information from confusing the model
	- uniform usage of names or terms is essential
- Labeling Quality
	- accurate labeling is crucial to avoid misleading th emodel

Diverse Representation
- Inclusion of varied demographics
	- have data from multiple demographic groups so that your model serves everyone
- Balanced data sources
	- don't just rely on one data source, have multiple such as news, social media, literature, etc.
- Regional and linguistic variety

Regular Updates
- new vocabulary and trends
- cultural and social norms
- model retraining
	- periodically updating the model with fresh data helps maintain alignment with knowledge and societal standards

- anonymize data when working with PII
- be transparent about data sources

# Creating an NLP Data Loader Lab

Data Set
- an object in PyTorch that represents a collection of data samples
- Each data sample consists of one or more input features and their corresponding target labels
- You can use your data set to transform your data as needed

Data Loader
- responsible for efficiently loading and batching data from a data set
- abstracts away the process of iterating over a data set, shuffling and dividng it into batches for training
- in NLP, data loader is used to process and transform your text data rather than just the data set
- Params for data loader
	- data set to load from
	- batch size (samples per batch)
	- shuffle: bool
- All data loaders in PyTorch are iterator
	- Iterator is an object that can be looped over
	- used to traverse large data sets without loading all elements into memory simultaneously
	- PyTorch data loader processes data in batches
- Data loaders can also be used for tokenizing, sequencing, converting samples to same size, transforming data into tensors

Custom Collate Function
- purpose is to prepare and format individual data samples into batches that can be efficiently processed by ML models

Process of creating a data loader:
1. Create a custom dataset that inherits the `torch.utils.data.dataset` class
	1. This custom dataset should also potentially tokenize and return a tensor built on the vocab of the dataset
	2. vocab might be a input to create the custom dataset
2. create a collate function to possibly pad the inputs for data processing
	1. Each batch of data needs to be the same size, but not all batches need to have the same size
	2. Can also tokenize and convert to tensors in the collate function
3. Load the data in by passing in the custom_dataset to the DataLoader as well as batch_size
4. The dataloader will have all the batches of tensors to loop through

# References
https://www.coursera.org/learn/generative-ai-llm-architecture-data-preparation/home/module/2