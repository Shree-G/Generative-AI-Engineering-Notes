## Overview
Generative Pre-trained Transformer Models:
- there exist many different types of GPTs
	- image->text
	- text->text
	- text->image
	- text->video
- In it's core, it trained to predict the next word in a sequence of words. If trained on a large enough dataset, it gets very good at predicting these next words.

High level preview:
- Tokenization: the input data is split up into "tokens", or small pieces
- Each token gets a vector of numbers, which represents it's place in a high dimensional space
	- words that are similar to each other are "close" together in this high dimensional space
- The sequence of vectors pass through what's called an attention block
	- this allows the vectors to talk to each other and "update" their values so that those vectors are more correct with regards to their actual meaning
	- a machine learning model and a fashion model both have model, but the word model means very different things.
	- basically a giant pile of matrix multiplications
- These vectors go through Multi-layer Perceptron/Feed forward layer
	- go through the same operation in parallel
	- asking a long list of questions to each vector, and updating vectors based on the "answers" to those questions
	- basically a giant pile of matrix multiplications
- Somehow the essential meaning of the passage has been baked into the very last vector in the sequence
	- We then perform an operation that produces a probability distribution of what token come next
- Feed this system a bit of "seed text", and repeat the above steps to generate more and more text
- To make a chatbot, just feed in a "system prompt": "What follows is a conversation between a user and a helpful, very knowledgable, AI assistant. User: {PROMPT} AI Assistant: ___"
	- The LLM basically predicts what an AI assistant would do next ... LMFAO

### Deep Learning Background
- finding models that scale very well with billions of parameters
- these parameters are basically little dials that tune themselves based on on an input and expected output
- data is usually represented as a list, matrix, or higher dimensional array called a "tensor" for input data, and output data is usually another list of real numbers
- the input data is progressively transformed in layers until you get to the final output
- these model parameters is called "weights" - due to weighted sums in a matrix vector product
	- the actual brain, the actual thing learnt during training
- almost all the actual calculation looks like matrix multiplication

## Attention is all you need!

Motivating Examples
- Attention is supposed to enable other embeddings to pass information to this one in the next step after word embeddings
	- potentially quite far away information, richer meaning
- A well-trained model should make it so that the transformations of an embedding force it closer to the actual meaning of the token in context of every other token
	- this happens through multiplications/additions of vectors
- Tower might point toward a meaning that is a large tall noun, but preceded by Eiffel, and you should get a meaning that is closer to France, or Paris. If that is preceded by miniature, then the vector should change to mean something close to collectibles, or hobbyists
- If you input in a long mystery novel, and end with the sentence "and the murderer was", then by the end of all these calculations, the very last vector (was) in the sequence should have enough information to predict the next word

The attention pattern
- There are some Q matrix that gets multiplied to every word, and a corresponding K matrix that gets multiplied to every word. This essentially asks a "question" and receives a "key" that answers that question. 
- Both of these get multiplied together to create a dot product, to measure similarity. 
- 

## Complete Process

### 1. Word Embeddings and tokenization
- for the purpose of this walkthrough, we're going to pretend that tokenization will be words, instead of what actually it is
- the first model is called the "Embedding matrix"
	- has a vector for all possible words, and basically just assigns each token to a vector that is in the embedding matrix
	- the position of the word is also encoded in this first step, with the embedding matrix, through complex calculations
	- the embedding matrix is trained by just starting out with completely random weights, and then using back propagation and a shit ton of input data to train it
- We literally embed the word in a very high dimensional space (GPT3 has 12288 dimensions)
- Words with similar semantic meaning usually fall within the same "region" in this very high dimensional space
- Curious findings
	- The vector that you get when you subtract king-queen is similar to the one that you get when you subtract man-woman, or uncle-aunt
	- E(Italy)-E(Germany) = E(Mussolini) - E(Hitler)
- 617 million weights
- As this input data continues to get transformed, the goal is for it to "soak" up meaning from words around it, to give it a rich context that can help it predict the next word more accurately
- context size: amount of input data you can have at a time
	- GPT2 context size was 2048
	- this is why bots sometime lose context or lose memory

### 5. Unembedding
- Use another matrix that maps the very last vector in that long list of transformations to a list of 50,000 values that represent the words that were in the initial embedding matrix
	- very last vector because each vector in the last massive matrix tries to predict the very next word, so we can just use the last vector to predict the unknown word
		- the values in that very last vector are called logits
	- This matrix is called the unembedding matrix
- Then use some kind of nonlinearity function to normalize it into probability distributions (softmax)
- same number of parameters as the embedding matrix

Softmax:
- the outputs that you get after unembedding do not follow a valid distribution rules
- softmax makes it so that the largest values end up close to 1, and smallest values end up closest to 0.
- e to the power of the each of the numbers by the sum of all of those values
- using softmax instead of normal max function because it is differentiable so possible to compute back propagation on it
- Temperature: when T is larger, we give more weight to lower values, and when T is smaller, bigger values get more weight
	- 0 means the max value always gets 100%
	- The higher it goes, the more weight the "smaller" values get

# References
https://www.youtube.com/watch?v=wjZofJX0v4M - Transformers, the tech behind LLMs - 3Blue1Brown
https://www.youtube.com/watch?v=eMlx5fFNoYc - Attention in transformers, step-by-step - 3Blue1Brown
