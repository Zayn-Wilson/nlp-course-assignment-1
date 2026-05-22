# NLP Course Assignments

This repository contains the code and deliverables for an NLP (Natural Language Processing) course. Each task folder corresponds to a specific assignment covering different NLP techniques.

## Repository Structure

```
nlp-course-assignment-1/
├── README.md
├── tut1-model.pt                          # Pre-trained Seq2Seq model
├── Task1_Word2Vec/                        # Word Embeddings with Gensim
├── Task2_CBOW/                            # Continuous Bag of Words
├── Task3_SurnameClassification/           # RNN-based Surname Classification
├── Task4_SurnameGeneration/               # RNN-based Surname Generation
├── Task5_AttentionVisualization/          # Attention Mechanism Visualization
├── Sequence_to_Sequence_LSTM/             # Sequence to Sequence with LSTM
└── data/                                  # Shared datasets
```

## Tasks Overview

### Task 1: Word2Vec with Gensim
**Task1_Word2Vec/**

Word embedding using Gensim's Word2Vec implementation.

- **Model**: Skip-Gram and CBOW
- **Dataset**: Restaurant reviews (train.csv)
- **Key Features**:
  - Word vector extraction
  - Semantic similarity calculation
  - Word analogies (e.g., "restaurant" + "party" - "quiet")

**Files**:
- `word2vec_gensim.ipynb` - Main implementation with Skip-Gram training
- `6.2.4 (1).py` - Reference code from course materials

---

### Task 2: Continuous Bag of Words (CBOW)
**Task2_CBOW/**

CBOW model implementation from scratch using PyTorch.

- **Model**: CBOW Classifier with embeddings and fully connected layers
- **Dataset**: Frankenstein novel dataset
- **Key Concepts**:
  - Vocabulary construction
  - Context-target pair generation
  - Mini-batch training

**Files**:
- `Continuous_Bag_of_Words_CBOW.py` - Main implementation
- `Continuous_Bag_of_Words_CBOW_gitee.py` - Alternative version

---

### Task 3: RNN Surname Classification
**Task3_SurnameClassification/**

Character-level RNN classifier for predicting nationality from surname.

- **Model**: LSTM/GRU-based classifier
- **Dataset**: `surnames_with_splits.csv` (18 nationalities)
- **Key Features**:
  - Character embeddings
  - Variable-length sequence processing
  - Multi-class classification

**Files**:
- `Task3_Surname_RNN.py` - Classification model
- `surnames_with_splits.csv` - Dataset

**Sample Output**:
```
Surname: McMahan -> Arabic (0.85)
Surname: Nakamoto -> Japanese (0.72)
```

---

### Task 4: RNN Surname Generation
**Task4_SurnameGeneration/**

Character-level RNN for generating surnames with optional nationality conditioning.

**Model 1: Unconditioned Generation**
- Generates arbitrary surnames without nationality control

**Model 2: Conditioned Generation**
- Uses nationality embedding to guide generation
- Produces country-specific surname styles

**Files**:
- `Model1_Unconditioned_Surname_Generation.py`
- `Model2_Conditioned_Surname_Generation.py`

**Sample Output**:
```
Unconditioned: Holand, Rossby, Nakamura
Conditioned (German): Schmidt, Weber, Muller
Conditioned (Russian): Ivanov, Petrov, Volkov
```

---

### Task 5: Attention Visualization
**Task5_AttentionVisualization/**

Visualization of attention weights from a pre-trained sentiment classifier.

- **Model**: DistilBERT fine-tuned on SST-2
- **Task**: Sentiment Analysis (Positive/Negative)
- **Visualization**:
  - Single head attention heatmap (Layer 3, Head 5)
  - Multi-layer overview (6 layers)
  - Token importance bar charts

**Files**:
- `attention_visualization.py` - Main code
- `pyproject.toml` - Dependencies
- `README.md` - Task documentation
- `attention_visualization/` - Output images

**Attention Patterns Observed**:
1. **Local dependencies** in early layers (syntax)
2. **Sentiment word focus** across all layers
3. **[CLS] token aggregation** for classification
4. **Long-range dependencies** in higher layers

---

### Extra: Sequence to Sequence Learning with LSTM
**Sequence_to_Sequence_LSTM/**

Seq2Seq model with LSTM encoder-decoder architecture.

- **Model**: LSTM-based Encoder-Decoder
- **Application**: Sequence generation/translation
- **Pre-trained weights**: `tut1-model.pt`

**Files**:
- `Sequence to Sequence Learning with LSTM.ipynb` - Jupyter notebook
- `tut1-model.pt` - Pre-trained model weights

---

## Dependencies

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
gensim>=4.0.0
pandas>=1.4.0
```

For Task 5 (Attention Visualization):
```bash
uv pip install torch transformers matplotlib seaborn
```

## Quick Start

```bash
# Clone repository
git clone git@github.com:Zayn-Wilson/nlp-course-assignment-1.git
cd nlp-course-assignment-1

# Task 1: Word2Vec
cd Task1_Word2Vec
jupyter notebook word2vec_gensim.ipynb

# Task 2: CBOW
cd Task2_CBOW
python Continuous_Bag_of_Words_CBOW.py

# Task 3: Surname Classification
cd Task3_SurnameClassification
python Task3_Surname_RNN.py

# Task 4: Surname Generation
cd Task4_SurnameGeneration
python Model1_Unconditioned_Surname_Generation.py
python Model2_Conditioned_Surname_Generation.py

# Task 5: Attention Visualization
cd Task5_AttentionVisualization
uv pip install torch transformers matplotlib seaborn
python attention_visualization.py
```

## Course Topics Covered

| Task | Topic | Key Technique |
|------|-------|---------------|
| 1 | Word Embeddings | Word2Vec (Skip-Gram/CBOW) |
| 2 | Context Modeling | CBOW Classifier |
| 3 | Sequence Classification | LSTM/GRU RNN |
| 4 | Sequence Generation | Conditional RNN |
| 5 | Model Interpretability | Attention Visualization |
| Extra | Seq2Seq | Encoder-Decoder LSTM |

## Author

Zayn Wilson

## License

MIT License
