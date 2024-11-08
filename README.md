# Multi-Language Named Entity Recognition (NER) Analysis

This repository contains a comprehensive Multi-Language NER analysis project utilizing XLM-RoBERTa for Named Entity Recognition tasks across multiple languages. The project demonstrates cross-lingual transfer capabilities and includes both fine-tuned and zero-shot models. It features a Gradio-based application for real-time NER predictions, making it interactive and user-friendly.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Understanding the Dataset](#understanding-the-dataset)
- [Data Preprocessing](#data-preprocessing)
- [Tokenizer Comparison: BERT vs XLM-R](#tokenizer-comparison-bert-vs-xlm-r)
- [Tokenizer for NER Analysis](#tokenizer-for-ner-analysis)
- [Model Metrics](#model-metrics)
- [Model Training](#model-training)
- [Cross-Lingual Transfer](#cross-lingual-transfer)
- [Zero-Shot vs Fine-Tuned Model](#zero-shot-vs-fine-tuned-model)
- [Gradio Application](#gradio-application)
- [Installation](#installation)
- [Usage](#usage)
- [Uploading to Hugging Face](#uploading-to-hugging-face)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project investigates the application of a multi-language transformer model, specifically **XLM-RoBERTa**, for Named Entity Recognition (NER) across languages. By fine-tuning on a German dataset and transferring the model to other languages such as French, Italian, and English, the project explores cross-lingual transfer in NER tasks.

## Dataset
The primary dataset for this project is the **XTREME** dataset, commonly used for benchmarking multi-lingual models. We specifically use the German, French, Italian, and English subsets for NER analysis. The dataset includes tokens, labels, and language identifiers.

## Understanding the Dataset
Each dataset entry consists of:
- **Tokens:** Individual words or subwords for token-level classification.
- **Labels:** Corresponding entity labels (e.g., Person, Organization, Location).
- **Language:** The language in which the data is written.

This section of the project inspects the distribution of labels and languages, ensuring a balanced dataset representative of multiple languages.

## Data Preprocessing
Preprocessing includes:
1. Converting entity labels into numerical representations.
2. Tokenizing text for model compatibility.
3. Splitting data into training, validation, and test sets based on language proportions.

## Tokenizer Comparison: BERT vs XLM-R
Comparing **BERT** and **XLM-RoBERTa** tokenizers reveals differences in how tokens are processed:
- BERT: Uses special tokens `[CLS]` and `[SEP]`.
- XLM-R: Uses language-specific encoding and subword segmentation with an underscore (`_`).

## Tokenizer for NER Analysis
The **XLM-RoBERTa tokenizer** is configured specifically for token classification in NER, allowing robust handling of multiple languages and entity labels.

## Model Metrics
Standard NER metrics are used for evaluation:
- **Precision, Recall, F1 Score:** To evaluate entity-level performance across languages.
- **Per-language metrics:** To assess cross-lingual transfer effectiveness.

## Model Training
The **XLM-RoBERTa** model is fine-tuned on German NER data. Key steps include:
- Setting up an NER-friendly architecture with token classification layers.
- Training with early stopping and monitoring loss for optimal performance.

## Cross-Lingual Transfer
After fine-tuning on German data, the model is tested on French, Italian, and English to observe cross-lingual transfer capabilities. This section analyzes performance in zero-shot settings.

## Zero-Shot vs Fine-Tuned Model
We assess the difference between zero-shot (direct application without fine-tuning) and fine-tuned models. Fine-tuning improves accuracy significantly across languages, demonstrating the benefits of transfer learning in NER tasks.

## Gradio Application
A Gradio-based web application allows users to interact with the NER model in real time. Users can input text in different languages, and the application will highlight recognized entities.

### Example Usage:
- Enter text in one of the supported languages.
- The application will display the entities identified, along with their labels.

## Installation
To run this project locally:
 **Additional dependencies** for Gradio:
    ```bash
    pip install gradio
    ```

 Install the Hugging Face Transformers library:
    ```bash
    pip install transformers
    ```

## Usage
1. **Run the Gradio application:**
    ```bash
    python gradio_app.py
    ```

2. **Model Fine-tuning and Evaluation:** Run the provided notebook for full training and evaluation steps.

## Uploading to Hugging Face
Once fine-tuned, the model can be uploaded to Hugging Face for easy sharing:

1. **Authenticate:**
    ```bash
    huggingface-cli login
    ```

2. **Upload the model:**
    ```python
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained("path_to_your_fine_tuned_model")
    model.push_to_hub("your_hf_username/your_model_name")
    ```

## Contributing
Contributions are welcome! Please fork this repository, create a new branch for your feature, and submit a pull request.

## License
This project is licensed under the MIT License.
