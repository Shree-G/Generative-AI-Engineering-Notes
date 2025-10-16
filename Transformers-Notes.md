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

### 2. Attention is all you need!

Motivating Examples
- Attention is supposed to enable other embeddings to pass information to this one in the next step after word embeddings
	- potentially quite far away information, richer meaning
- A well-trained model should make it so that the transformations of an embedding force it closer to the actual meaning of the token in context of every other token
	- this happens through multiplications/additions of vectors
- Tower might point toward a meaning that is a large tall noun, but preceded by Eiffel, and you should get a meaning that is closer to France, or Paris. If that is preceded by miniature, then the vector should change to mean something close to collectibles, or hobbyists
- If you input in a long mystery novel, and end with the sentence "and the murderer was", then by the end of all these calculations, the very last vector (was) in the sequence should have enough information to predict the next word

The attention pattern:
- There is some W_Q matrix that gets multiplied to every word vector, and a corresponding W_K matrix that gets multiplied to every word vector. This essentially asks a "question" and receives a "key" that answers that question. This is represented by Q_1... and K_1...
	- W_Q is learnt from the model through back-propagation - filled with tunable weights
	- The Q vector and K vector are much smaller, 128 dimensions
- Both of these get multiplied together to create a dot product, to measure similarity. 
- When a Q and K are similar enough based on dot-product, you think of them as "matching" - this is a 2 dim space
	- The embeddings of "key" word vector (K_2) "attend to" the embeddings of a "query" word vector (Q_4)
	- These dot products are any value from -inf to inf
	- However, we want these values to add up to 1 and act as weights - so we use softmax
	- Also divide by root of dimension of the key-query space for numerical stability
- After you apply softmax, you will have normalized values for each combination of Q_x and K_y.
	- This directly explains how relevant the "key" word vector is to the "query" word vector
- This grid is called an **attention pattern**
	- The attention pattern's size scales based on the SQUARE of the context size, making scaling context size non-trivial
- During the training process, it's useful to have the model try to predict the next token for every possible next token, and then reward or punish the model. This makes it so that the model has a lot more training data to go off of
	- **Masking**: This means that you never want later tokens to influence earlier tokens, which means you want the values of the attention pattern where they represent later tokens influencing earlier ones, to be 0 once softmax is applied. Therefore, we set them to -inf before softmax is applied, effectively turning them into 0 when softmax is applied

Updating Embeddings for better meanings:
- we use a third matrix, called the value matrix (W_V) and multiply it by the embedding of the first word (K_y), and we add the result to the value of the second word (Q_x). This encodes some level of meaning that K_y carries into Q_x
	- For example, if K_y is fluffy, and Q_x is creature, then Q_x's meaning will now represent something closer to a fluffy creature
	- Value matrix is usually split up into the product of two matrices to reduce the number of parameters
		- Low rank transformation
		- Value matrix is split up into two - and one of the value matrices are put together in one huge matrix called the **output matrix**. This is for mathematical efficiency.
- Can think of multiplying the value matrix by the embedding of the word as saying: if this word is relevant to adjusting the meaning of another word, what exactly should be added to the vector of the other word in order to reflect this?
- multiply the Value Matrix by all of the embeddings for Keys, and you will get a value vector
- then you multiply that value vector by the corresponding weight associated with the Key-Query pair, and add all the rescaled weights to the original Query vector.
	- you can think of this as the weights determining how much each word should affect each other word
- Do this for all the Key-Value pairs, and you will have a more refined list of vectors that should better represent the meanings of the word with the context around it
- This whole process is called: **One head of attention**
	- ~6.3 million parameters for one head of attention
	- This is specifically called self-attention

Scaling:
 -  For every different "query" that you ask from one word to another, you need a completely different set of W_Q, W_K, W_V to capture the different attention patterns
	 - This means each separate head of attention in an attention block has different weight matrices to ask different "queries"
	 - In practice, the true behavior of these weights are essentially unknown - these weights just do whatever it takes to arrive at the best next word
- In each attention block, there are multiple heads of attention. GPT3, for example, has 96 heads of attention per attention block
- Each of these 96 distinct heads of attention output 96 different "variations" of updates for each token in the input
	- You sum together all of those proposed changes, one for each head, and add those all together to the original embedding, to then arrive at your "better" meaning embedding for that token
- By running multiple different heads in parallel, you are giving the model multiple different ways of learning many different ways of how context changes meaning
- Each block ends up with 600 million parameters.
- There are many attention blocks, which gives the hope that each better embedding will influence other better embeddings, over and over again until the tokens are influenced with a shit ton of contextual meaning
- There are 96 different layers of this, bringing the total up to ~58 billion parameters
- This operation is really nice because it is extremely parallelizable - using GPUs makes scaling better

### 3. Multilayer perceptrons

#### Overview:
- input is a sequence of vectors from the end of an attention block
- these input vectors will go through a series of calculations, and the ending will be a vector with the same dimension as the input vector
- This new vector will get added to the initial vector, to form the final output of the MLP layer
- This is all parallelized for all of the vectors in the input

#### First Step:
- First step is to multiply the initial vector by a matrix that has been pre-trained with a bunch of weights, similar to other matrices we've seen so far
- Every row can be thought of as asking some kind of probing question to the initial vector
	- This is because we're doing a dot-product and a dot-product essentially measures the similarity/parallelness between two vectors
	- For example, it can be thought of as asking "is this vector representing something relating to a 4 legged animal?" - if it is, then it would be closer to 1, if it is not, then it is closer to 0
- A bias (with pre-trained tunable weights) - a vector of the same dimension of the matrix multiplication output, is also added to this matrix multiplication
- The number of row of this matrix is can be considered to be the number of questions the model is asking the input data
	- In GPT3, the number of rows is 49152 - 4 times the amount of the embedding space (cleaner for hardware)

#### Second Step:
- We need some level of nonlinearity in between, so that the model can approximate some level of how human language is oftentimes very nonlinear
	- nonlinearity also helps with drawing separation boundaries for decisions - such as curved boundaries
	- without nonlinearity, this whole operation would be equivalent to a single linear transformation, which doesn't help it as much
- The nonlinear function GPT uses is ReLU - which effectively makes negative values 0 and leaves nonnegative values as is.
	- this can be thought of as ignoring all the "no's" to the "question" asked in the previous stage, and only caring about the sure yesses
- In most actual models, we use a more smooth function called GeLU
- The **"neurons"** of a transformer, we talk about the outputs of the nonlinearity function
	- we would say the neuron is active whenever the value is positive, and inactive whenever the value is 0
	- Very similar to an AND gate

#### Third Step:
- We again have another matrix that essentially transforms this higher dimensional vector back into the input dimension
	- It's useful to think of the matrix column by column
	- Each column represents a bunch of "directions" that we can add - whether we add them is conditional on whether the neuron itself is added
- We also have another bias vector that we add
- Both of these are pre-trained parameter based matrices and vectors

#### Final Step and bookkeeping
- Add the vector from the previous step to the input vector
- In essense, we will have added on "extra" information based on the questions we have asked the vector
	- This step might add on a "basketball" dimension to a vector that already represented Michael Jordan
- This is a process happening in parallel across ALL the input tokens, making the size of the parameters not 50,000, but 50,000 * num of input tokens
- There are 96 of these layers, so around ~120 billion parameters

### 4. Unembedding
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
HQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8 - How LLMs might store facts - 3Blue1Brown
