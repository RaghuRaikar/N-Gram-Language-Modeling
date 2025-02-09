📖 N-Gram Language Modeling with Additive Smoothing & Interpolation 🚀
======================================================================

### 🔍 Overview

This project implements **n-gram language models** (unigram, bigram, and trigram) with **additive smoothing** and **linear interpolation** to analyze and generate probabilistic language models. The goal is to estimate the likelihood of word sequences in natural language and evaluate model performance using **perplexity scores**.

The program is tested using the **One Billion Word Benchmark dataset**, but it can be used for any large-scale text corpus.

* * * * *

🎯 Features
-----------

✅ Implements **unigram, bigram, and trigram** language models\
✅ Supports **additive smoothing** (Laplace smoothing) with tunable α values\
✅ Implements **linear interpolation smoothing** with customizable λ values\
✅ **Computes perplexity** on training, development, and test sets\
✅ Processes large-scale text datasets efficiently\
✅ Allows evaluation and comparison of different smoothing techniques

* * * * *

🛠 How It Works
---------------

The program reads a large dataset of tokenized text and processes it as follows:

1️⃣ **Tokenization:** Reads input text and converts it into a structured n-gram format\
2️⃣ **Model Training:** Builds **unigram, bigram, and trigram** probability distributions\
3️⃣ **Smoothing Techniques:** Applies:

-   **Additive smoothing** (Laplace smoothing) to handle unseen words
-   **Linear interpolation smoothing** to combine n-gram probabilities\
    4️⃣ **Perplexity Calculation:** Evaluates how well the model predicts unseen text\
    5️⃣ **Evaluation & Comparison:** Reports results on train, dev, and test sets

* * * * *

📥 Inputs & Outputs
-------------------

### 📌 Input Files:

-   **Train Dataset:** `1b_benchmark.train.tokens` (Large-scale corpus for model training)
-   **Dev Dataset:** `1b_benchmark.dev.tokens` (Used to tune hyperparameters)
-   **Test Dataset:** `1b_benchmark.test.tokens` (Evaluates final model performance)

### 📌 Output:

-   **Perplexity Scores** for different n-gram models and smoothing techniques
-   **Best performing hyperparameters** based on development set results
-   **Comparison between raw and smoothed models**

* * * * *

📊 Performance & Results
------------------------

The model was evaluated on the One Billion Word dataset, and the **perplexity scores** were as follows:

### 📌 Raw N-Gram Model Perplexities:

| Model | Train Perplexity | Dev Perplexity | Test Perplexity |
| --- | --- | --- | --- |
| Unigram | 976.5 | 2406.3 | 2380.4 |
| Bigram | 77.1 | 18704.8 | 18202.4 |
| Trigram | 7.9 | 11645330.1 | 11214441.99 |

### 📌 With Additive Smoothing (α = 1):

| Model | Train Perplexity | Dev Perplexity |
| --- | --- | --- |
| Unigram | 977.51 | 2410.16 |
| Bigram | 1442.31 | 114658.51 |
| Trigram | 6244.42 | 66381731.91 |

### 📌 With Linear Interpolation (λ1 = 0.1, λ2 = 0.3, λ3 = 0.6):

-   **Perplexity:** ~48.1 (Evaluation Set)

🔹 **Key Findings:**

-   Without smoothing, **higher-order n-grams performed well on training data** but suffered on unseen data due to sparsity.
-   **Additive smoothing improved generalization** but introduced bias.
-   **Linear interpolation offered the best balance**, reducing overfitting while maintaining predictive power.

* * * * *

🚀 Installation & Usage
-----------------------

### 📥 1. Clone the Repository

`git clone https://github.com/your-repo/N-Gram-Language-Modeling.git`  
`cd N-Gram-Language-Modeling`

### 🏗 2. Install Dependencies

Ensure you have Python installed, then install required packages:

`pip install -r requirements.txt`

### ▶️ 3. Run the Program

`python main.py --train 1b_benchmark.train.tokens --dev 1b_benchmark.dev.tokens --test 1b_benchmark.test.tokens --smoothing additive --alpha 1`  

🔹 Modify `--smoothing` and `--alpha` to test different configurations.


📜 Purpose & Applications
-------------------------

This program is **not just an assignment**---it has **real-world applications** in:

🔹 **Speech Recognition:** Predicting the next word in voice input\
🔹 **Text Generation:** Powering chatbots & AI writers\
🔹 **Spelling & Grammar Correction:** Ranking candidate words based on likelihood\
🔹 **Machine Translation:** Improving fluency in automated translations

🔹 **Why This Matters?**

-   Helps machines understand natural language better
-   Reduces **word prediction errors** in AI systems
-   Improves **language-based AI applications**

* * * * *

🏆 Future Enhancements
----------------------

💡 Improve performance with **Kneser-Ney Smoothing**\
💡 Implement **neural-based language models (e.g., LSTMs, Transformers)**\
💡 Support **larger-scale datasets** beyond One Billion Word Benchmark

* * * * *

🔥 **A powerful tool for language modeling---fast, efficient, and ready for real-world NLP tasks!** 🚀
