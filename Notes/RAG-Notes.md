What is RAG:
- process of optimizing the output of a large language model
- references an authoritative knowledge based outside of its training data source
- train an LLM of an organization's internal knowledge base without needing to retrain the model
- Motivation: The alternative would be to fine tune the model, but the model is HUGE with billions of parameters, so we want to come up with another way: RAG

How does RAG work (data ingestion pipeline + retrieval pipeline):
- Have an external **vector** database with the authoritative data
	- will help you save data in vectors inside the database
- We build a data ingestion pipeline to put our authoritative data into the vector database
	- Data -> Parsing -> Embedding -> Vector DB
		- This data can be in any format - problem
		- Parsing: Reading the data and figuring out how to chunk the data up
		- Embedding Models: converting text to vectors (a numerical representation for text)
	- This is to apply some kind of similarity search so that relevant information can be retrieved from the Vector DB
	- We have created a **knowledge base** that doesn't exist within the LLM
- Now, when a user gives a query, the query will first be:
	- Embedded (converted to a vector)
	- similarity search is applied with the data in the vector db, which will return some relevant context
- We use this context with a specific non-user prompt, that we will send to the LLM.
- We get an output from the pipeline based on the authoritative data in the vector DB

LangChain Document Structure:
- page_content(str)
	- the actual content of the page
- metadata(dict)
	- some metadata that will be useful to have when referencing the document in the future
	- when you are applying similarity search, you can apply filters based on the metadata
- This format is very important because we will be using this format to push this to the vector db
- LangChain Document Loaders:
	- pdfLoader
	- csvLoader
	- WebBaseLoader
	- etc
- These loaders will give you the output in the form of a document structure

# More advanced theoretical concepts:

## Query Translation with Re-written queries

### Multi-Query
- Motive: 
	- Sometimes user queries are poorly worded, and due to this get embedded ambiguously. 
	- Embedding long documents is hard. 
	- Once the poorly worded question is embedded, it becomes really hard to improve search no matter what you do later in the pipeline
- You can take a question and make it less or more abstract in an effort to get better results
- Intuitively, we want to transform a question into multiple perspectives
- We can combine with parallelized retrieval with multiple versions of the same question

How it actually works
- We ask an LLM to give us multiple versions of the user's input prompt
- Then we retrieve chunks based on each version of the input prompt
- Then we combine a unique superset of all of the chunks and feed it into the LLM as context
- serialize and deserialize
	- serialize:
		- convert a document to a string
		- use the lanchain.load dumps function
	- deserialize
		- convert a string to a Document
		- use the langchain.load loads function
	- used to make the documents "hashable" which allows you to use a set to easily find and remove duplicates
### RAGFusion
- It's the same as multi-query, the only difference is we rank the chunks that get outputted with a reciprocal rank function
- this is very useful is we want to control the number of chunks that we give the LLM for context - if we only want 3, then we can just take the top 3 ranked chunks

How it actually works
- We ask an LLM to give us multiple versions of the user's input prompt
- Then we retrieve chunks based on each version of the input prompt
- Then we **rank and combine** a unique superset of all of the chunks and feed it into the LLM as context

## Query Translation with Sub-questions
### Decomposition
- Ask the LLM to break the initial question down into multiple sub-questions
- Tackle all the sub-questions sequentially, and pass in the output of the first sub-question to the next sub-question, and so on and so forth
- By the end, you should have a context with all of the sub-questions for the last sub-question.

### Answer Individually
- Exactly like Decomposition except for a few differences
- Instead of passing in the output of prior subquestions to future ones, we just answer each question individually and the concatenate the answer
- Relevant for problems where the initial prompt contains set of sub-problems where the problems don't depend on each other or aren't related

## Query Translation with more abstraction

### Step Back
- Asking the LLM to give a more generalized question that the LLM can use as context before answering the actual question
- Examples of Step-Back questions:
```python
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
```
- You actually pass these questions as examples to the LLM, as well as pass your original user question and ask it to create a step back question.
- You retrieve chunks for both the step back question and the actual question, dedupe them, and the give it to the LLM with a system prompt for a final answer.
- This could be really convenient for domains with very conceptual knowledge based questions to automatically formulate higher level questions

### HyDE
- motivation: 
	- documents are usually long, while questions are usually very short
	- this might cause the question to be embedded in such a way that it's not close to relevant documents in the high dimensional space
- to circumvent this, we extrapolate the question into a document, in hopes that it will be closer to documents that are more relevant to it.
![[Pasted image 20251027224846.png]]
- this can overcome the challenges of inaccurate retrieval for some domains where there is some knowledge already available on answering specific questions

## Routing
- Routing that question based on the content of the question to:
	- the relevant data source
	- the correct prompt to use
	- any other conditional step in the process
Logical Routing:
- We give an LLM knowledge of the various data sources/prompts
- then we let the LLM choose which data source/prompt to use for that specific question
- We essentially need to constrain the llm to give us a structured output like so:
```python
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt 
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router 
router = prompt | structured_llm
```
- we can essentially pass a class with a specified output to the LLM, and have it return us an object of that output
- The llm.with_structured_output(RouteQuery) method does something special:
	1. It tells the LLM, "You **must** respond by calling the `RouteQuery` function/tool."
	2. To do this, it **automatically converts your `RouteQuery` class into a detailed "function-calling" schema** and includes it in the _real_ system prompt sent to the model.

Semantic Routing:
- we conduct a similarity search on the input question (embedded) and the available prompts (embedded) that we want to use
- choose to use the prompt with the highest similarity score

## Query Construction
- using function calling to search for specific fields in the metadata itself
- ex: 
	- i ask for research papers published by stanford after 2020.
	- pass a function in that sets "publisher" : "stanford" and "date-published" : 2020>
	- searches the vector-db for documents only corresponding to those conditions
- very convenient as it does meta-data filtering **on the fly**. this is very powerful if you are going to get very structured questions that are metric based
- how to do this
	- basically just create a class with optional/required fields and descriptions for each of those fields
	- pass it into a structured llm call

## Indexing

### Multi-Representation Indexing
- you take a document, and you distill it in some way, and you embed that distillation which also has a index to a document store where all the documents live
- this distillation/summary might contain a "crisper", more useful version of a potentially longer document
- you use this summary to perform similarity search on the embedded question
- once you get the most similar distillations, you then use the attached index to attach the entire document to that llm call
- this is specifically useful for long-context llms, as you can give it way more information this way

## Random Shit to Remember:
tiktoken_encoder:
- tiktoken_encoder is the same encoder that OpenAI uses
- it measures chunks using tokens rather than characters. 
- this is much better than using characters because it aligns with the context windows and pricing of LLMs
	- Ex: 
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)
```

.map method will take the list of whatever and run the method for each one in parallel
- Ex:
```python
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
```

example of rag chain:
```python
final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)
```
- Here's what that first block `{"context": ..., "question": ...}` actually does:
	- **It's a `RunnableMap`:** This dictionary-like structure is a component that runs multiple things in parallel. When you call `final_rag_chain.invoke({"question": "..."})`, this `RunnableMap` is the first step.
	- **It receives the _original_ input:** The input `{"question": "..."}` is passed to _both_ keys inside the map.
	- **`"context": retrieval_chain`:** This part says: "Take the original input (`{"question": "..."}`), send it to the `retrieval_chain`, get the result (the list of documents), and put that result into a _new_ key called `context`."
	- **`"question": itemgetter("question")`:** This part says: "Take the original input (`{"question": "..."}`), use `itemgetter` to pluck out the value for the `question` key, and pass that value along under the key `question`."
		- RunnablePassThrough() will do the same thing
		- Use `RunnablePassthrough` if the input is the raw value you want to pass (e.g., a string).
    
- Use `itemgetter("key")` if the input is a dictionary and you want to pass one of its values.

Good example of human ai chatbot conversation format creation to send to llm:
```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
```
- interesting thing is that in LangChain it essentially looks like human chatting with AI format, so this is how you format it for the AI for previous responses
- `ChatPromptTemplate` is for standard chat prompts, and `FewShotChatMessagePromptTemplate` is for building prompts that include a few examples.
	- you must use ChatPromptTemplate.from_template() to create a prompt object that can be piped into
- A `system` message sets the LLM's role and task.

Using custom/lambda functions as a part of your LCEL chain:
```python
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)
```
- use RunnableLambda to pass a function (or lambda function)

# References
https://www.youtube.com/watch?v=o126p1QN_RI
https://www.youtube.com/watch?v=sVcwVQRHIc8&t=4s
https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_5_to_9.ipynb