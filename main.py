import math  # Import math for mathematical functions
from collections import Counter  # Import defaultdict and Counter from collections for enhanced dictionary and counting functionalities

class Ngram():
    def __init__(self, smoothing=0):
        self.unigram_probabilities = {}
        self.bigram = {}
        self.bigram_prob = {}
        self.trigram_counts = {}  # Store counts of each trigram
        self.trigram_prob = {}  # Probabilities of each trigram
        self.smoothing = smoothing # smoothing value

    def fit(self, sentences, vocab, bigram_model, model_type):
        if model_type == 'unigram':
            total_tokens = sum(vocab.values())
            vocab_size = len(vocab)
            self.unigram_probabilities = {word: (count + self.smoothing) / (total_tokens + self.smoothing * vocab_size) for word, count in vocab.items()}
        elif model_type == 'bigram':
            vocab_size = len(vocab)
            for sentence in sentences:
                tokens = ['<START>'] + sentence
                for i in range(len(tokens) - 1):
                    bigram = (tokens[i], tokens[i+1])
                    self.bigram[bigram] = self.bigram.get(bigram, 0) + 1

            for bigram, count in self.bigram.items():
                first_word = bigram[0]
                denominator = len(sentences) if first_word == "<START>" else vocab.get(first_word, 1)
                self.bigram_prob[bigram] = (count + self.smoothing) / (denominator + self.smoothing * vocab_size)
        elif model_type == 'trigram':
            vocab_size = len(vocab)
            for sentence in sentences:
                tokens = ['<START>', '<START>'] + sentence
                for i in range(2, len(tokens)):
                    trigram = (tokens[i-2], tokens[i-1], tokens[i])
                    self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1

            for trigram, count in self.trigram_counts.items():
                if trigram[:2] == ('<START>', '<START>'):
                    self.trigram_prob[trigram] = (count + self.smoothing) / (len(sentences) + self.smoothing * vocab_size)
                else:
                    bigram_count = bigram_model.bigram.get(trigram[:2], 1)
                    self.trigram_prob[trigram] = (count + self.smoothing) / (bigram_count + self.smoothing * vocab_size)

    def perplexity(self, sentences, model):
        if model == 'unigram':
            log_prob_sum = sum(math.log2(self.unigram_probabilities.get(token, 1e-10))
            for sentence in sentences for token in sentence)
            token_count = sum(len(sentence) for sentence in sentences)
            return 2 ** (-log_prob_sum / token_count)
        elif model == 'bigram':
            log_prob_sum = 0
            total_bigrams = 0
            for sentence in sentences:
                tokens = ['<START>'] + sentence
                for i in range(len(tokens) - 1):
                    bigram = (tokens[i], tokens[i+1])
                    prob = self.bigram_prob.get(bigram, 1e-10)
                    log_prob_sum -= math.log2(prob)
                    total_bigrams += 1
            return float('inf') if total_bigrams == 0 else 2 ** (log_prob_sum / total_bigrams)
        elif model == 'trigram':
            log_prob_sum = 0
            total_trigrams = 0
            for sentence in sentences:
                tokens = ['<START>', '<START>'] + sentence
                for i in range(2, len(tokens)):
                    trigram = (tokens[i-2], tokens[i-1], tokens[i])
                    prob = self.trigram_prob.get(trigram, 1e-10)  # Use a tiny value to prevent log(0)
                    log_prob_sum += -math.log2(prob)
                    total_trigrams += 1
            return float('inf') if total_trigrams == 0 else 2 ** (log_prob_sum / total_trigrams)

def tokenize_sentences(data):
    """Tokenize each line in the data and append '<STOP>' to each tokenized list."""
    return [line.strip().split() + ['<STOP>'] for line in data]


def create_vocabulary(data):
    """Create vocabulary with word counts and replace infrequent tokens with '<UNK>'."""
    # Count occurrences of each token
    token_counts = Counter(token for sentence in data for token in sentence)
    
    # Replace tokens occurring less than 3 times with '<UNK>'
    modified_data = [[token if token_counts[token] >= 3 else '<UNK>' for token in sentence] for sentence in data]

    # Update the token counts to include only tokens that appear at least 3 times
    # Ensure that '<STOP>' is always included if it was part of the original data
    vocab = {token: count for token, count in token_counts.items() if count >= 3 or token == '<STOP>'}
    vocab['<UNK>'] = sum(1 for sentence in modified_data for token in sentence if token == '<UNK>')

    # Output the number of unique tokens in the modified data
    print("Vocabulary size:", len(vocab))

    return modified_data, vocab


def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:  # Open the file in read mode
        data = file.readlines()  # Read all lines from the file
    return data  # Return the list of lines

def main():
    # Uncomment to print command-line arguments (not used here)
    # print(args)
    train_data = read_file("1b_benchmark.train.tokens")  # Read training data
    tokenized_train_data = tokenize_sentences(train_data)  # Tokenize training data
    test_data = read_file("1b_benchmark.dev.tokens")  # Read test data
    tokenized_test_data = tokenize_sentences(test_data)  # Tokenize test data
    train, V = create_vocabulary(tokenized_train_data)  # Build vocabulary from training data
    unigram = Ngram(smoothing=0.6)  # Instantiate a unigram model
    bigram = Ngram(smoothing=0.6)  # Instantiate a bigram model
    trigram = Ngram(smoothing=0.)  # Instantiate a trigram model
    unigram.fit(train,V,bigram,'unigram')  # Fit the unigram model on the training data
    bigram.fit(train, V, bigram, 'bigram')  # Fit the bigram model on the training data
    trigram.fit(train, V, bigram, 'trigram')  # Fit the trigram model using the bigram model

    # Print perplexity values for the unigram, bigram, and trigram models on training and test data
    print("Train Perplexity Unigram: " + str(unigram.perplexity(train, 'unigram')))
    print("Train Perplexity Bigram: " + str(bigram.perplexity(train, 'bigram')))
    print("Train Perplexity Trigram: " + str(trigram.perplexity(train, 'trigram')))
    print("Test Perplexity Unigram: " + str(unigram.perplexity(tokenized_test_data, 'unigram')))
    print("Test Perplexity Bigram: " + str(bigram.perplexity(tokenized_test_data, 'bigram')))
    print("Test Perplexity Trigram: " + str(trigram.perplexity(tokenized_test_data, 'trigram')))
    lambda_values = [
        (0.1, 0.3, 0.6),  # Specified in the assignment
        (0.3, 0.3, 0.4),
        (0.2, 0.2, 0.6),
        (0.4, 0.3, 0.3),
        (0.5, 0.3, 0.2)
    ]
    
    best_lambda = None
    best_dev_perplexity = float('inf')
    
    for lambdas in lambda_values:
        lambda1, lambda2, lambda3 = lambdas
        train_perplexity = unigram.linear_interpolation(train, lambda1, lambda2, lambda3, unigram, bigram, trigram)
        dev_perplexity = unigram.linear_interpolation(tokenized_dev_data, lambda1, lambda2, lambda3, unigram, bigram, trigram)
        print(f"Train Perplexity with λ1={lambda1}, λ2={lambda2}, λ3={lambda3}: {train_perplexity}")
        print(f"Dev Perplexity with λ1={lambda1}, λ2={lambda2}, λ3={lambda3}: {dev_perplexity}")
        if dev_perplexity < best_dev_perplexity:
            best_dev_perplexity = dev_perplexity
            best_lambda = lambdas
    
    # Report test set perplexity using the best hyperparameters from development set
    best_lambda1, best_lambda2, best_lambda3 = best_lambda
    test_perplexity = unigram.linear_interpolation(tokenized_test_data, best_lambda1, best_lambda2, best_lambda3, unigram, bigram, trigram)
    print(f"Test Perplexity with best λ1={best_lambda1}, λ2={best_lambda2}, λ3={best_lambda3}: {test_perplexity}")
    
    # Experiment with using half the training data
    half_train_data = tokenized_train_data[:len(tokenized_train_data) // 2]
    half_train, half_V = create_vocabulary(half_train_data)
    unigram_half = Ngram(smoothing=0)
    bigram_half = Ngram(smoothing=0)
    trigram_half = Ngram(smoothing=0)
    unigram_half.fit(half_train, half_V, bigram_half, 'unigram')
    bigram_half.fit(half_train, half_V, bigram_half, 'bigram')
    trigram_half.fit(half_train, half_V, bigram_half, 'trigram')
    half_train_perplexity = unigram_half.linear_interpolation(half_train, best_lambda1, best_lambda2, best_lambda3, unigram_half, bigram_half, trigram_half)
    half_dev_perplexity = unigram_half.linear_interpolation(tokenized_dev_data, best_lambda1, best_lambda2, best_lambda3, unigram_half, bigram_half, trigram_half)
    print(f"Half Train Perplexity: {half_train_perplexity}")
    print(f"Half Dev Perplexity: {half_dev_perplexity}")
    
    # Experiment with changing the threshold for <UNK> conversion to 5
    train_unk5, V_unk5 = create_vocabulary(tokenized_train_data, unk_threshold=5)
    unigram_unk5 = Ngram(smoothing=0)
    bigram_unk5 = Ngram(smoothing=0)
    trigram_unk5 = Ngram(smoothing=0)
    unigram_unk5.fit(train_unk5, V_unk5, bigram_unk5, 'unigram')
    bigram_unk5.fit(train_unk5, V_unk5, bigram_unk5, 'bigram')
    trigram_unk5.fit(train_unk5, V_unk5, bigram_unk5, 'trigram')
    unk5_train_perplexity = unigram_unk5.linear_interpolation(train_unk5, best_lambda1, best_lambda2, best_lambda3, unigram_unk5, bigram_unk5, trigram_unk5)
    unk5_dev_perplexity = unigram_unk5.linear_interpolation(tokenized_dev_data, best_lambda1, best_lambda2, best_lambda3, unigram_unk5, bigram_unk5, trigram_unk5)
    print(f"Train Perplexity with <UNK> threshold 5: {unk5_train_perplexity}")
    print(f"Dev Perplexity with <UNK> threshold 5: {unk5_dev_perplexity}")

if __name__ == '__main__':
    main()  # Execute the main function if the script is run as a standalone program
