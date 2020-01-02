############################################################
# Undergraduate Research - John Zhang



# Imports
############################################################

import numpy as np

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import contextlib
import sys

############################################################
# Helper Functions
############################################################

def loadTrainingSentences(file_path):
    commandTypeToSentences = {}

    with open(file_path, 'r') as fin:
        for line in fin:
            line = line.rstrip('\n')
            if len(line.strip()) == 0 or "##" == line.strip()[0:2]:
                continue
            commandType, command = line.split(' :: ')
            commandType = commandType[:-9]
            if commandType not in commandTypeToSentences:
                commandTypeToSentences[commandType] = [command]
            else:
                commandTypeToSentences[commandType].append(command)

    return commandTypeToSentences

# The following structure is useful for supressing print statements
class PrintsToOuterSpace(object):
    def write(self, x): pass

@contextlib.contextmanager
def noStdOut():
    save_stdout = sys.stdout
    sys.stdout = PrintsToOuterSpace()
    yield
    sys.stdout = save_stdout

############################################################
# Intent Detection
############################################################

def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2))

class Bert:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        self.storedResults = None

        self.wordEmbeddings = {}

    def generateWordEmbeddings(self):
        sent = "[CLS] Turn your lights off. [SEP]"
        word = "off"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Turn your lights on. [SEP]"
        word = "on"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Set your lights to maximum intensity. [SEP]"
        word = "maximum"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Set your lights to minimum intensity. [SEP]"
        word = "minimum"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Increase the blue value of your back LED by 50%. [SEP]"
        word = "increase"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Decrease the blue value of your back LED by 50%. [SEP]"
        word = "decrease"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Waddle off. [SEP]"
        word = "off"
        self.wordEmbeddings["off2"] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Waddle on. [SEP]"
        word = "on"
        self.wordEmbeddings["on2"] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Turn your back light green. [SEP]"
        word = "back"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Turn your front light green. [SEP]"
        word = "front"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)


        sent = "[CLS] Blink your logic display. [SEP]"
        word = "blink"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Dim your lights holoemitter. [SEP]"
        word = "dim"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

        sent = "[CLS] Increase your speed by 50 percent. [SEP]"
        word = "percent"
        self.wordEmbeddings[word] = self.wordEmbeddingNewSentence(sent, word)

    def wordEmbeddingNewSentence(self, sent, word):
        tokenized_text = self.tokenizer.tokenize(sent)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        sum_vec = torch.sum(token_embeddings[tokenized_text.index(word)][-4:], dim = 0).numpy()
        return sum_vec

    def contextWordSim(self, word, sentWord):
        if sentWord not in self.tokenized_text: return 0 # bert tokenization works differently, for common words they are all in

        sum_vec = torch.sum(self.token_embeddings[self.tokenized_text.index(word)][-4:], dim = 0).numpy()

        if word == "off": return max(cosineSimilarity(self.wordEmbeddings["off"], sum_vec), cosineSimilarity(self.wordEmbeddings["off2"], sum_vec))
        if word == "on": return max(cosineSimilarity(self.wordEmbeddings["on"], sum_vec), cosineSimilarity(self.wordEmbeddings["on2"], sum_vec))
        if word in self.wordEmbeddings: return cosineSimilarity(self.wordEmbeddings[word], sum_vec)

        print("This world has not had a contextualized word embedding yet. See the method generateWordEmbeddings in the Bert class of r2d2_bert.py for more details.")
        return 0

    def bertSentenceEncode(self, sentences):
        if type(sentences) is not list: sentences = [ sentences ]

        returnedEmbeddings = []

        for sentence in sentences:
            marked_text = "[CLS] " + sentence + " [SEP]"

            self.tokenized_text = self.tokenizer.tokenize(marked_text) # Tokenize our sentence with the BERT tokenizer.
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenized_text)
            segments_ids = [1] * len(self.tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

            # the following is only needed for creating word embeddings, but better to do it here instead of individually for each word
            token_embeddings = torch.stack(encoded_layers, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            self.token_embeddings = token_embeddings.permute(1,0,2)

            # Use the second to last bert layer for word embeddings
            token_vecs = encoded_layers[11][0]

            # Calculate the average of all token vectors in the sentence.
            sentence_embedding = torch.mean(token_vecs, dim=0)

            returnedEmbeddings.append(sentence_embedding.numpy())

        return np.array(returnedEmbeddings)

    def sentenceToEmbeddings(self, commandTypeToSentences):
        '''Returns a tuple of sentence embeddings and an index-to-(sentence, category)
        dictionary.

        Inputs:
            commandTypeToSentences: A dictionary in the form returned by
            loadTrainingSentences. Each key is a string '[category]' which
            maps to a list of the sentences belonging to that category.

        Let m = number of sentences.
        Let n = dimension of vectors.

        Returns: a tuple (sentenceEmbeddings, indexToSentence)
            sentenceEmbeddings: A mxn numpy array where m[i:] containes the embedding
            for sentence i.

            indexToSentence: A dictionary with key: index i, value: (sentence, category).
        '''
        indexToSentence = {}

        i = 0
        for category in commandTypeToSentences:
            sentences = commandTypeToSentences[category]
            for sentence in sentences:
                indexToSentence[i] = (sentence, category)
                i += 1

        sentenceEmbeddings = self.bertSentenceEncode([indexToSentence[j][0] for j in range(i)])

        return sentenceEmbeddings, indexToSentence

    def createSentenceEmbeddings(self, file_path):
        commandTypeToSentences = loadTrainingSentences(file_path)
        sentenceEmbeddings, indexToSentence = self.sentenceToEmbeddings(commandTypeToSentences)
        self.storedResults = (sentenceEmbeddings, indexToSentence, commandTypeToSentences)

        return commandTypeToSentences

    def getCategory(self, sentence):
        '''Returns the supposed category of 'sentence'.

        Inputs:
            sentence: A sentence

            file_path: path to a file containing r2d2 commands

        Returns:
            a string 'command', where 'command' is the category that the sentence
            should belong to.
        '''
        commandEmbedding = self.bertSentenceEncode(sentence)

        sentenceEmbeddings, indexToSentence, commandTypeToSentences = self.storedResults

        sortList = []

        for i in range(sentenceEmbeddings.shape[0]):
            similarity = cosineSimilarity(commandEmbedding, sentenceEmbeddings[i, :])
            sortList.append((i, similarity))

        similarSentences = sorted(sortList, key = lambda x: x[1], reverse = True)

        closestSentences = [x[0] for x in similarSentences]

        commandDict = {}
        for category in commandTypeToSentences:
            commandDict[category] = 0

        commandDict[indexToSentence[closestSentences[0]][1]] += 1
        commandDict[indexToSentence[closestSentences[1]][1]] += 0.5
        commandDict[indexToSentence[closestSentences[2]][1]] += 0.5
        commandDict[indexToSentence[closestSentences[3]][1]] += 0.2
        commandDict[indexToSentence[closestSentences[4]][1]] += 0.2

        print(commandDict)
        print("Closest sentence was: " + indexToSentence[closestSentences[0]][0])
        print(cosineSimilarity(commandEmbedding, sentenceEmbeddings[closestSentences[0], :]))

        if cosineSimilarity(commandEmbedding, sentenceEmbeddings[closestSentences[0], :]) < 0.73:
            return "no"

        return max(commandDict, key=commandDict.get)