# evaluate_sim_2.py

import numpy as np
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer  # Install rouge-score if not available
from nltk.util import ngrams

# Ensure the necessary NLTK packages are downloaded
nltk.download('wordnet')

# Increase text size for plots
plt.rcParams.update({'font.size': 14})

class Evaluator:
    def __init__(self, retrieved_contexts, generated_answers):
        self.retrieved_contexts = retrieved_contexts  # True context (used as expected answers)
        self.generated_answers = generated_answers  # Model-generated answers

    def evaluate_f1_score(self):
        y_true = [1] * len(self.generated_answers)
        y_pred = [1 if answer in self.retrieved_contexts else 0 for answer in self.generated_answers]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return precision, recall, f1

    def evaluate_bleu(self):
        scores = [
            sentence_gleu([context.split()], answer.split()) 
            for context, answer in zip(self.retrieved_contexts, self.generated_answers)
        ]
        return np.mean(scores)

    def evaluate_meteor(self):
        scores = [meteor_score([context.split()], answer.split()) for context, answer in zip(self.retrieved_contexts, self.generated_answers)]
        return np.mean(scores)

    def evaluate_gleu(self):
        scores = [sentence_gleu([context.split()], answer.split()) for context, answer in zip(self.retrieved_contexts, self.generated_answers)]
        return np.mean(scores)

    def evaluate_rouge(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rougeL_scores = []

        for context, answer in zip(self.retrieved_contexts, self.generated_answers):
            scores = scorer.score(context, answer)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return np.mean(rouge1_scores), np.mean(rougeL_scores)

    def evaluate_cosine_similarity(self):
        vectorizer = CountVectorizer().fit_transform(self.retrieved_contexts + self.generated_answers)
        vectors = vectorizer.toarray()

        cosine_scores = []
        for i in range(len(self.generated_answers)):
            cosine = cosine_similarity([vectors[i]], [vectors[len(self.retrieved_contexts) + i]])[0][0]
            cosine_scores.append(cosine)

        return np.mean(cosine_scores)

    def evaluate_jaccard(self):
        def jaccard_similarity(list1, list2):
            s1 = set(list1)
            s2 = set(list2)
            return len(s1.intersection(s2)) / len(s1.union(s2))

        scores = [
            jaccard_similarity(context.split(), answer.split())
            for context, answer in zip(self.retrieved_contexts, self.generated_answers)
        ]
        return np.mean(scores)

    def evaluate_all(self):
        precision, recall, f1 = self.evaluate_f1_score()
        bleu = self.evaluate_bleu()
        meteor = self.evaluate_meteor()
        gleu = self.evaluate_gleu()
        rouge1, rougeL = self.evaluate_rouge()
        cosine_sim = self.evaluate_cosine_similarity()
        jaccard_sim = self.evaluate_jaccard()

        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'BLEU Score': bleu,
            'METEOR Score': meteor,
            'GLEU Score': gleu,
            'ROUGE-1 Score': rouge1,
            'ROUGE-L Score': rougeL,
            'Cosine Similarity': cosine_sim,
            'Jaccard Similarity': jaccard_sim,
        }

    def visualize_results(self, evaluation_results):
        self.plot_evaluation_results(evaluation_results)
        self.plot_metric_trends(evaluation_results)

    def plot_evaluation_results(self, evaluation_results):
        metrics = list(evaluation_results.keys())
        scores = list(evaluation_results.values())

        plt.figure(figsize=(15, 9))
        plt.barh(metrics, scores, color='skyblue')
        plt.xlabel('Score')
        plt.title('Evaluation Metrics for Generated Answers')
        plt.xlim(0, 1)  # Assuming scores are between 0 and 1
        plt.grid(axis='x')
        plt.show()

    def plot_metric_trends(self, evaluation_results):
        metrics = list(evaluation_results.keys())
        scores = np.array(list(evaluation_results.values())).reshape(1, -1)

        plt.figure(figsize=(15, 9))
        plt.plot(metrics, scores.flatten(), marker='o', color='orange', linewidth=10)
        plt.xticks(rotation=45)
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.title('Evaluation Metric Trends')
        plt.ylim(0, 1)  # Assuming scores are between 0 and 1
        plt.subplots_adjust(bottom=0.2)
        plt.grid()
        plt.show()

# Function to load CSV and perform evaluations
def load_data_and_evaluate(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Extract relevant columns (assuming CSV columns are Context, Question, Actual Answer)
    retrieved_contexts = df['Context'].tolist()  # Context (the true context or reference answer)
    generated_answers = df['Actual Answer'].tolist()  # Generated answers from the model

    # Initialize Evaluator
    evaluator = Evaluator(retrieved_contexts, generated_answers)

    # Perform evaluations
    evaluation_results = evaluator.evaluate_all()
    print("Evaluation Results:", evaluation_results)

    # Visualize the results
    evaluator.visualize_results(evaluation_results)

# Example Usage
if __name__ == "__main__":
    csv_file = "ground_truth_1.csv"  # Replace with the actual CSV file path
    load_data_and_evaluate(csv_file)
