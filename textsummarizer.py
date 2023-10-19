import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx

def read_article(text):
    
    sentences = []
    
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]"," ")

    return sentences

def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    #build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1
    
    #build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
            
    return 1-cosine_distance(vector1,vector2)

# Create similarity matrix among all sentences
def build_similarity_matrix(sentences,stop_words):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
                
    return similarity_matrix


# Generate and return text summary
def generate_summary(text,top_n):
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    stop_words = stopwords.words('english')
    summarize_text = []
    
    # Step1: read text and tokenize
    sentences = read_article(text)
    
    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    
    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    
    # Step 5: get the top n number of sentences based on rank    
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
    
    # Step 6 : output the summarized version
    return " ".join(summarize_text),len(sentences)




# For console output
'''
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx

def read_article(text):
    
    sentences = []
    
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]"," ")

    return sentences

def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    #build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1
    
    #build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
            
    return 1-cosine_distance(vector1,vector2)

# Create similarity matrix among all sentences
def build_similarity_matrix(sentences,stop_words):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
                
    return similarity_matrix


# Generate and return text summary
def generate_summary(text,top_n):
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    stop_words = stopwords.words('english')
    summarize_text = []
    
    # Step1: read text and tokenize
    sentences = read_article(text)
    print('Sentence Tokenization:\n', sentences, end = '\n\n')
    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    print('Similarity Matrix:', sentence_similarity_matrix, end='\n\n')
    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    print('Similarity Graph:', sentence_similarity_graph, end='\n\n')
    scores = nx.pagerank(sentence_similarity_graph)
    print('Scores:', scores, end='\n\n')
    
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    
    # Step 5: get the top n number of sentences based on rank    
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
    
    # Step 6 : output the summarized version
    print("Summarized Text: ".join(summarize_text),len(sentences))

generate_summary('The concept of artificial intelligence (AI) has rapidly evolved over the years. AI, or the simulation of human intelligence by machines, has garnered significant attention in recent times. AI technologies, such as natural language processing and computer vision, are being integrated into various applications. From virtual assistants like Siri and Alexa to self-driving cars, AI is changing the way we live and work. Some of the pioneers in the field, like Alan Turing and John McCarthy, laid the foundation for AI research. Machine learning, a subset of AI, involves algorithms that allow computers to learn from data and make predictions. Deep learning, a more advanced form of machine learning, has led to breakthroughs in image and speech recognition. Ethical concerns about AI, including job displacement and privacy, are being widely discussed. Governments and organizations are investing heavily in AI research to stay competitive in the global market. The development of AI has the potential to revolutionize industries like healthcare and finance. As AI continues to progress, the possibilities seem limitless, and its impact on society remains a topic of fascination and debate.',3)
'''