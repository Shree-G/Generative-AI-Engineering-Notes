## Generative AI Architecture
### Significance of Generative AI

Understanding Generative AI

- Generative AI involves deep learning models that can create various types of content, such as text, images, and audio, based on the data they have been trained on.
- These models learn patterns and structures from existing data to generate new, relevant outputs, similar to how an artist creates original work.

Types of Generative AI Models

- Text Generation: Models like GPT can generate coherent text, predict story continuations, and translate languages while maintaining context.
- Image Generation: Models such as DALL-E and GAN can create images from text prompts or modify existing images, allowing for creative applications like deep fakes.
- Audio Generation: Models like WaveNet can produce realistic speech and facilitate text-to-speech synthesis.

Applications in Various Industries

- Generative AI is utilized for automating content creation, enhancing accessibility through natural language translation, and improving customer support with chatbots.
- Specific applications include analyzing medical images in healthcare, making financial predictions, enhancing gaming experiences, and generating artificial data for IT and data science.

### Generative AI Architectures and Models

Generative AI Architectures

- Recurrent Neural Networks (RNNs) utilize sequential data and a loop-based design, allowing them to remember previous inputs for tasks like language modeling.
- Transformers employ a self-attention mechanism to efficiently process information, enabling parallelization and improved accuracy in tasks such as translation.

Generative Models

- Generative Adversarial Networks (GANs) consist of a generator and a discriminator, working in a competitive manner to create realistic samples, particularly useful in image and video generation.
- Variational Autoencoders (VAEs) use an encoder-decoder framework to learn patterns in data, generating new samples that share similar characteristics.

Diffusion Models and Training Approaches

- Diffusion models generate images by learning to remove noise from distorted examples, relying on statistical properties of training data.
- Different training approaches are highlighted: RNNs use loops, transformers focus on self-attention, GANs engage in competition, VAEs learn characteristics, and diffusion models depend on statistical properties.

### Generative AI for NLP

Generative AI Architectures

- Generative AI enables machines to understand and generate human-like responses, enhancing language processing through context awareness.
- The evolution began with rule-based systems, progressed to machine learning, and advanced significantly with deep learning and transformer architectures.

Applications of Generative AI in NLP

- Significant advancements have been made in machine translation, chatbots, sentiment analysis, and text summarization.
- Generative AI improves accuracy in translations, creates more natural chatbot interactions, enhances sentiment analysis, and provides precise text summaries.

Large Language Models (LLMs)

- LLMs, such as GPT and BERT, are foundational models trained on vast datasets, enabling them to understand and generate language effectively.
- These models have billions of parameters and can be fine-tuned for specific tasks, but they may also produce inaccurate information and require careful consideration of biases.

### Overview of Libraries and Tools
- There are various libraries and tools that you can use to develop NLP applications using generative AI. Some tools are PyTorch, TensorFlow, Hugging Face, LangChain, and Pydantic. 
- PyTorch is an open source deep learning framework. It is a Python-based library well-known for its ease of use, flexibility, and dynamic computation graphs. 
- TensorFlow is an open-source framework for machine learning and deep learning. It provides tools and libraries to facilitate the development and deployment of machine learning models. 
	- The tight integration of TensorFlow with Keras provides a user-friendly high-level neural networks API, facilitating rapid prototyping and building and training deep learning models. 
- Hugging Face is a platform that offers an open-source library with pretrained models and tools to streamline the process of training and fine-tuning generative AI models. It offers libraries such as Transformers, Datasets, and Tokenizers. 
- LangChain is an open-source framework that helps streamline AI application development using LLMs. It provides tools for designing effective prompts. 
- Pydantic is a Python library that helps you streamline data handling. It ensures the accuracy of data types and formats before an application processes them.

### Exploring Generative AI Libraries Lab

#### Neural Structures behind Generative AI

The state of the art technology now is Transformers, which LLMs are built off of. LLMs:
	- have lots of neurons that learn from huge amounts of text
	- not good at understanding the bigger context or the meaning of words

Before we had transformers, we had other methods for making computers generate text:
- N-gram language models
	- predict what words come next in a sentence based on the words that came before
- RNNs
	- maintains a 'memory' or 'hidden state' throughout the sequence by employing loops
	- enables RNNs to capture temporal dependencies in sequential data
	- processes one word at a time, updating the hidden state at each word. by the end of the sentence, it has valuable information about the context of the sentence itself.
- LSTM (Long short term memory) and gated recurrent units (GRUs)
	- advanced versions of RNNs
	- struggled with long sequences and long-term dependencies
- Seq2seq models w/ attention
	- built with RNNs or LSTMs
	- designed to handle tasks like translation where input sequence is transformed into an output sequence
	- attention mechanism allowed model to focus on relevant parts of the input sequence when generating output, improving performance

First coded chatbot interaction LOL:
```
You:  what is the sun's age

Chatbot: The sun is about to set on the horizon. It's going to be a beautiful day.

You:  quit

Chatbot: Goodbye!
```
- this was with `facebook/blenderbot-400M-distill`

Second coded chatbot interaction...
```
You:  what is the age of the sun

Chatbot: 365 days

You:  are you sure

Chatbot: are you sure?

You:  fuck, you got me there.
```
- this was with google/flan-t5-base

Third coded chatbot interaction - these are getting funnier and funnier
```
You:  what is the age of the sun

Chatbot: 20th century

You:  are you sure

Chatbot: no

You:  valid.

Chatbot: I agree with the sentence.
```
- this was with google/flan-t5-small