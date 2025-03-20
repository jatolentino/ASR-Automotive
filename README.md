# ASR ML Model for the Automotive Industry
## Overview
This is a machine learning model designed to recognize spoken terms with various English accents within the automotive industry, developed from scratch.

## 1. Synthetic Data creation
Go to the directory `syntehtic_data` and follow the steps below:
```sh
cd synthetic_data
```
### 1.1. Tools
- MeloTTS
- PyTorch
- NLTK
- TensorFlow

### 1.2. Steps
- Requirements
    - conda `>= 24.9.1`
    - python `>= 3.10`

- Create a conda environment and activate it
```sh
conda melotts
conda activate melotts
```

- Install pytorch
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install MeloTTS
```sh
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
conda create -n melotts python=3.10
pip install -e .
python -m unidic download
```

- Install NLTK
```sh
pip install nltk
```

>**Possible errors with nltk and solution:**
- Create a folder called `nltk_data` inside the `synthetic_data\MeloTTS` directory
    ```
    import nltk
    nltk.download('punkt')  # Downloads the tokenizer models
    nltk.download('averaged_perceptron_tagger')  # Downloads the POS tagger
    nltk.download('stopwords')  # Downloads the stop words corpus
    nltk.download('wordnet')  # Downloads the WordNet corpus
    nltk.data.path.append(r'path_to_nltk_data')
    ```

- Error with importing nltk and `averaged_perceptron_tagger_eng`:
    - Open a python shell and do:
    ```python
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    exit()
    ```

### 1.3. Run a simple test

```sh
python main.py
```

This will generate a wav file called output