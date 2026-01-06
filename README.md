# Semantic Similarity NLP

A comprehensive Natural Language Processing project that explores semantic similarity using multiple embedding techniques and transformer models. This project demonstrates document preprocessing, word embeddings, and semantic similarity measurements using various state-of-the-art NLP models.

## 📋 Project Overview

This project implements and compares different approaches to measure semantic similarity between texts:

1. **Document Preprocessing with spaCy**
2. **Traditional Embeddings** (Word2Vec, FastText)
3. **TF-IDF Baseline**
4. **Transformer-based Embeddings** (BERT)

## 🎯 Features

- **PDF Text Extraction**: Automated extraction and preprocessing of text from research papers
- **Advanced Text Preprocessing**: Tokenization, lemmatization, POS tagging, and named entity recognition using spaCy
- **Multiple Embedding Models**:
  - Word2Vec (Skip-gram architecture)
  - FastText (with subword information)
  - TF-IDF vectorization
  - BERT (all-MiniLM-L6-v2)
- **Semantic Similarity Measurement**: Cosine similarity comparison across different models
- **Model Performance Comparison**: Side-by-side evaluation of embedding techniques

## 🛠️ Technologies Used

- **Python 3.x**
- **spaCy** - Industrial-strength NLP
- **Gensim** - Word2Vec and FastText implementations
- **PyPDF2** - PDF text extraction
- **Sentence Transformers** - BERT embeddings
- **scikit-learn** - TF-IDF and similarity metrics
- **NumPy** - Numerical computations

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/BabarKhan4/Semantic-Similarity-NLP.git
cd Semantic-Similarity-NLP
```

2. Install required packages:
```bash
pip install spacy PyPDF2 gensim scikit-learn sentence-transformers numpy
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## 🚀 Usage

1. **Prepare your document**: Place your PDF file (e.g., `paper.pdf`) in the project directory

2. **Run the notebook**: Open and execute `MAIN_TP_2.ipynb` in Jupyter Notebook or VS Code

3. **Follow the workflow**:
   - Load and preprocess the PDF document
   - Train Word2Vec and FastText models
   - Compare semantic similarity using different approaches
   - Analyze results and model performance

## 📊 Project Structure

```
Semantic-Similarity-NLP/
│
├── MAIN_TP_2.ipynb          # Main Jupyter notebook with all implementations
├── README.md                 # Project documentation
└── paper.pdf                 # Sample research paper (add your own)
```

## 🔍 How It Works

### 1. Document Preprocessing
- Extracts text from PDF documents
- Cleans and normalizes text using regular expressions
- Processes text with spaCy for linguistic analysis
- Filters tokens based on POS tags (keeps nouns, verbs, adjectives, adverbs)
- Removes stopwords and non-alphabetic tokens

### 2. Embedding Training
- **Word2Vec**: Learns word representations based on surrounding context
- **FastText**: Extends Word2Vec with subword information for better handling of rare words

### 3. Similarity Measurement
Compares sentence pairs using multiple approaches:
- **TF-IDF + Cosine Similarity**: Traditional term-frequency approach
- **Word2Vec Embeddings**: Context-aware word vectors
- **FastText Embeddings**: Subword-aware embeddings
- **BERT Embeddings**: Contextual transformer-based representations

### 4. Performance Comparison
Evaluates and compares similarity scores across all models to determine which approach best captures semantic meaning.

## 📈 Sample Results

Example comparison for two semantically similar sentences:

| Model      | Similarity Score |
|------------|------------------|
| TF-IDF     | 0.XXX           |
| Word2Vec   | 0.410           |
| FastText   | 0.720           |
| BERT       | 0.XXX           |

*Note: FastText typically performs better due to its ability to capture subword patterns*

## 🎓 Learning Objectives

This project covers:
- ✅ PDF text extraction and preprocessing
- ✅ spaCy for advanced NLP tasks
- ✅ Training custom Word2Vec and FastText models
- ✅ Understanding word embeddings and vector representations
- ✅ Implementing semantic similarity measures
- ✅ Comparing traditional vs transformer-based approaches
- ✅ Named Entity Recognition (NER)
- ✅ Part-of-Speech (POS) tagging

## 📝 Tasks Completed

### Task 1: Document Preprocessing & Embedding Models
- [x] PDF text extraction with PyPDF2
- [x] Text preprocessing with spaCy
- [x] Word2Vec and FastText model training
- [x] Sentence similarity calculation
- [x] Named entity recognition
- [x] Most common words analysis

### Task 2: Model Comparison & Evaluation
- [x] TF-IDF baseline implementation
- [x] BERT embedding generation
- [x] Performance comparison across models
- [x] Similarity metrics evaluation

## 🔗 References

- [Word Embedding Survey Paper](https://arxiv.org/pdf/2003.07278)
- [spaCy Documentation](https://spacy.io/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Sentence Transformers](https://www.sbert.net/)

## 👤 Author

**Babar Khan**
- GitHub: [@BabarKhan4](https://github.com/BabarKhan4)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⭐ Show your support

Give a ⭐️ if this project helped you!
