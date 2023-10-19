# NLP-Extractive-Text-Summarizer
_An Extractive Text Summarizer Web-App using NLP in Python_

## Table of Contents
- [Scope](#scope)
- [Working](#working)
- [Setup](#setup)
- [GUI](#gui)

## Scope
1.	Generates an extractive summary of the corpus of text in user-defined no. of lines.
2.	Handles errors related to invalid, missing, or erroneous input.
3.	Appealing web-based GUI made using HTML, and CSS, with back-end connectivity provided via Python with Flask.

## Working
### 1.	Importing Required Libraries
This section imports the necessary libraries, such as NLTK, NumPy, and NetworkX. NLTK is used for natural language processing tasks, NumPy is used for numerical operations, and NetworkX is used to create and manipulate graphs.
 

### 2.	read_article(text) 
This function takes the input text and tokenizes it into sentences using NLTK's sent_tokenize function. It also performs some basic text preprocessing by removing non-alphanumeric characters from each sentence.

Given input: 
“The concept of artificial intelligence (AI) has rapidly evolved over the years. AI, or the simulation of human intelligence by machines, has garnered significant attention in recent times. AI technologies, such as natural language processing and computer vision, are being integrated into various applications. From virtual assistants like Siri and Alexa to self-driving cars, AI is changing the way we live and work. Some of the pioneers in the field, like Alan Turing and John McCarthy, laid the foundation for AI research. Machine learning, a subset of AI, involves algorithms that allow computers to learn from data and make predictions. Deep learning, a more advanced form of machine learning, has led to breakthroughs in image and speech recognition. Ethical concerns about AI, including job displacement and privacy, are being widely discussed. Governments and organizations are investing heavily in AI research to stay competitive in the global market. The development of AI has the potential to revolutionize industries like healthcare and finance. As AI continues to progress, the possibilities seem limitless, and its impact on society remains a topic of fascination and debate.”
 

### 3.	sentence_similarity(sent1, sent2, stopwords)
This function calculates the cosine similarity between two sentences, sent1 and sent2. It uses a bag-of-words model to represent sentences as vectors and then computes the cosine similarity between these vectors. The stopwords parameter is a list of words to be ignored in the comparison.

### 4.	build_similarity_matrix(sentences, stop_words)
This function creates a similarity matrix that represents the similarity between all pairs of sentences in the input text. It iterates through each pair of sentences and calculates their similarity using the sentence_similarity function. The result is stored in a matrix.
 

### 5.	generate_summary(text, top_n)
This is the main function that generates the text summary. It first reads and tokenizes the input text, then builds a similarity matrix for the sentences. Next, it constructs a sentence similarity graph using NetworkX, where nodes represent sentences, and edges represent the similarity between sentences. PageRank algorithm is applied to rank the sentences based on their importance in the graph. Finally, the top ‘top_n’ sentences are selected as the summary, and the summary is returned as a string.
 
## Setup
1. Install and set-up Python in any IDE.
2. Create a virtual environment using the command:
```python
python -m venv <venv-name>
```
3. Activate the virtual environment using the command:
```python
<venv-name>\Scripts\activate
```
5. Install the necessary libraries such as:
```python
pip install numpy
```
```python
pip install scipy
```
```python
pip install flask
```
```python
pip install nltk
```
```python
pip install networkx
```
6. Run app.py file:
```python
python app.py
```

## GUI
![image](https://github.com/ParthJaju/NLP-Extractive-Text-Summarizer/assets/81647051/58939137-f6e3-4d73-ada1-f2d6c1be8020)

### Input Screen
![image](https://github.com/ParthJaju/NLP-Extractive-Text-Summarizer/assets/81647051/8b7ebe06-d6b9-470f-a424-16d16b350dff)

### Results Screen
![image](https://github.com/ParthJaju/NLP-Extractive-Text-Summarizer/assets/81647051/618e7200-5eef-4873-8493-dea8e3d19d86)
 


