# Enhancing TinyBERT for Financial Sentiment Analysis Using GPT-Augmented FinBERT Distillation

## Project Overview
This repository contains the source code and resources for the research project [Enhancing TinyFinBERT for Financial Sentiment Analysis Using GPT-Augmented FinBERT Distillation.](https://arxiv.org/abs/2409.18999) This study explores the use of large language models (LLMs) like GPT-4 Omni for generating synthetic, domain-specific training data to enhance the performance of transformer-based models in financial sentiment analysis.

## Key Objectives
- **Enhance [FinBERT](https://huggingface.co/ProsusAI/finbert) using [GPT-4 Omni](https://openai.com/index/hello-gpt-4o/) Augmented Data** 
- **Test Augmented FinBERT's Generalization on Unseen Financial Data** 
- **Implement Knowledge Distillation from Augmented FinBERT to TinyFinBERT using unlabelled and labelled data created by [GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5-turbo) and GPT-4 Omni**
- **Evaluate and Compare the Performance of TinyFinBERT**
- **Test TinyFinBERT’s Generalization on Unseen Financial Data** 

## Methodology
This project leverages GPT-4 Omni for dataset augmentation to improve FinBERT's performance as a teacher model. Through a structured knowledge distillation process, we transfer insights from FinBERT to a smaller TinyFinBERT model, aiming for a reduced model footprint while retaining high performance.

### Overview of Methodology
- **Synthetic Data Generation:** Used LLMs such as GPT-4 Omni and GPT-3.5 Turbo to generate domain-specific synthetic training data, especially for mislabeled and complex financial statements. This approach enhances the FinBERT model's coverage and robustness.
- **Two-Tiered Knowledge Distillation:** Developed TinyFinBERT using a structured, dual-layer knowledge distillation approach with logit matching and intermediate layer alignment for robust performance transfer.

### Synthetic Data Generation
To address data sparsity and labeling challenges:
- **GPT-4 Omni** was employed for generating labeled data, focusing on accurate financial sentiment annotations.
- **GPT-3.5 Turbo** was used for cost-effective generation of unlabeled data, increasing diversity and handling rare financial statement structures.

### Knowledge Distillation
FinBERT, enriched with synthetic data, serves as the teacher model in knowledge distillation, guiding TinyFinBERT’s learning:
- **Logit Matching** aligns TinyFinBERT's outputs with FinBERT's predictions.
- **Intermediate Layer Transfer** encourages feature alignment across hidden layers, promoting TinyFinBERT’s retention of relevant knowledge.

### Model Training and Evaluation
The training process integrates both original and synthetic datasets, with models evaluated across diverse benchmarks:
- **Datasets Used:** Financial PhraseBank, FiQA 2018 Task1, and Forex News Annotated datasets were selected to assess model performance and generalization to real-world financial contexts.

### Advanced Training Techniques
For optimal model refinement and stability:
- **Discriminative Fine-Tuning** and **Slanted Triangular Learning Rates** were applied to enhance learning dynamics.
- **Gradual Unfreezing** prevents catastrophic forgetting, ensuring TinyFinBERT retains critical financial insights during finetuning.


## Datasets
- [**PhraseBank Dataset**](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)
- [**FiQA 2018 Task 1 Dataset**](https://sites.google.com/view/fiqa)
- [**Forex News Annotated Dataset**](https://doi.org/10.5281/zenodo.7976208)

These datasets were instrumental in training and evaluating TinyFinBERT, ensuring the model's effectiveness in real-world financial sentiment analysis tasks.

## Results
TinyFinBERT achieves performance comparable to its teacher model, FinBERT, with the advantage of reduced size and computational needs, demonstrating the effectiveness of LLMs in enhancing smaller models through advanced data augmentation and distillation techniques.

### Performance Comparison
The following tables display TinyFinBERT's performance compared to both the baseline FinBERT and the intermediary Augmented FinBERT across three distinct datasets: Financial PhraseBank (FPB Test), Forex, and FiQA. 

#### Table 1: Performance Results for TinyFinBERT vs. FinBERT and Augmented FinBERT
| Model               | Dataset   | Accuracy | F1 Score | Precision | Recall |
|---------------------|-----------|----------|----------|-----------|--------|
| FinBERT             | FPB Test  | 0.8423   | 0.8439   | 0.8545    | 0.8423 |
|                     | Forex     | 0.4801   | 0.4449   | 0.4988    | 0.4801 |
|                     | FIQA      | 0.5265   | 0.5563   | 0.6642    | 0.5265 |
| **Augmented FinBERT** | FPB Test | **0.8742** | **0.8739** | **0.8743** | **0.8742** |
|                     | Forex     | **0.4950** | **0.4797** | **0.5081** | **0.4950** |
|                     | FIQA      | **0.6217** | **0.6385** | **0.6709** | **0.6217** |
| TinyBERT            | FPB Test  | 0.1330   | 0.0329   | 0.6103    | 0.1330 |
|                     | Forex     | 0.3095   | 0.1463   | 0.0958    | 0.3095 |
|                     | FIQA      | 0.0925   | 0.0157   | 0.0086    | 0.0925 |
| **TinyFinBERT**     | FPB Test  | 0.8330   | 0.8330   | 0.8333    | 0.8330 |
|                     | Forex     | 0.4775   | 0.4572   | 0.4923    | 0.4775 |
|                     | FIQA      | 0.5660   | 0.5944   | 0.6560    | 0.5660 |

#### Table 2: Comparison of TinyFinBERT Performance as a Percentage of FinBERT
| Model               | Dataset   | Accuracy | F1 Score | Precision | Recall |
|---------------------|-----------|----------|----------|-----------|--------|
| **TinyFinBERT**     | FPB Test  | 98.90%   | 98.71%   | 97.52%    | 98.90% |
|                     | Forex     | 99.46%   | 102.76%  | 98.70%    | 99.46% |
|                     | FIQA      | 107.50%  | 106.85%  | 98.77%    | 107.50% |

The performance tables demonstrate that TinyFinBERT retains a high level of accuracy and F1 score across all tested datasets, closely matching FinBERT in most metrics and even surpassing it in some cases, all while **being significantly smaller (7.5 times smaller) than FinBERT.** This confirms the efficacy of our knowledge distillation approach and highlights TinyFinBERT’s viability as a more efficient, domain-specific language model for financial sentiment analysis.

## Model Parameter Analysis
This section provides an in-depth look at the configurations and parameter counts for each model variant used in this project.

### FinBERT
The **FinBERT** model, available on Hugging Face under [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), is based on the BERT-base architecture. FinBERT consists of:
- **12 Transformer encoder layers**
- **Hidden size**: 768
- **Feedforward size**: 3072
- **Attention heads**: 12
- **Parameter count**: ~110 million

A dense layer processes the output from the `[CLS]` token, commonly used in BERT models to aggregate sequence-level representations, making FinBERT particularly effective for classification tasks within financial text domains.

### Augmented FinBERT
**Augmented FinBERT** builds on the base FinBERT model, retaining the same architecture with:
- **12 Transformer encoder layers**
- **Hidden size**: 768
- **Feedforward size**: 3072
- **Attention heads**: 12
- **Parameter count**: ~110 million

Augmented FinBERT undergoes fine-tuning with advanced techniques:
- **Slanted Triangular Learning Rates**: Allows for dynamic learning rate adjustment across training stages.
- **Discriminative Fine-Tuning**: Assigns different learning rates to model layers based on their position.
- **Gradual Unfreezing**: Unfreezes model layers progressively to stabilize learning.

These strategies enhance Augmented FinBERT’s performance and robustness in handling nuanced financial text.

### TinyBERT
**TinyBERT**, hosted on Hugging Face under [huawei-noah/TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D), is a smaller, efficient BERT model pre-trained on a general corpus. TinyBERT’s architecture includes:
- **4 Transformer layers**
- **Hidden size**: 312
- **Feedforward size**: 1200
- **Attention heads**: 12
- **Parameter count**: ~14.5 million

This compact structure reduces computational demands while retaining sufficient representational power for many NLP tasks.

### TinyFinBERT
**TinyFinBERT** shares the same architecture as TinyBERT and incorporates FinBERT’s financial domain-specific knowledge through a targeted distillation process. TinyFinBERT has:
- **4 Transformer layers**
- **Hidden size**: 312
- **Feedforward size**: 1200
- **Attention heads**: 12
- **Parameter count**: ~14.5 million

By distilling knowledge from Augmented FinBERT, TinyFinBERT achieves strong performance in financial sentiment analysis with a significantly smaller footprint.

## TinyFinBERT Distillation Process Overview

### Method
- **Distillation Technique**: Adopts the Transformer distillation method from [Jiao et al., 2019](https://arxiv.org/abs/1909.10351), designed for transferring knowledge from a larger teacher network (N layers) to a smaller student network (M layers).
- **Layer Mapping**: Utilizes a layer correspondence mapping `n = g(m)`, where the m-th student layer learns from the g(m)-th teacher layer.

### Objective Function
- **Distillation Objective**: Defined as:
<img src="https://miro.medium.com/v2/resize:fit:974/1*6pHKhJ7HpOuKjpXObtERoA.png" width="371" height="237" alt="Knowledge Distillation Objective Function">

### Distillation Components
- **Embedding-layer Distillation**: Similar to hidden states, using MSE with a transformation matrix `W_e` for embeddings.
- **Attention-based Distillation**: Focuses on multi-head attention matrices using mean squared error (MSE) between the student's and teacher's attention matrices.
- **Hidden States-based Distillation**: Involves distilling the hidden states of Transformer layers with a learnable transformation matrix `W_h` for alignment.
- **Prediction-layer Distillation**: Utilizes soft cross-entropy loss to align the student’s and teacher's logits, adjusted by a temperature scaling factor `t`.
- **Unified Distillation Loss**: Consolidates the distillation objectives across different layers to ensure cohesive and targeted knowledge transfer.

### Process Overview and Data Usage
- **Datasets Used**: Financial Phrasebank dataset along with augmented data from GPT-3.5T and GPT-4o.
- **Teacher Model**: Augmented FinBERT.
- **Student Model (TinyFinBERT)**: Mirrors TinyBERT with 4 Transformer layers, a hidden size of 312, a feedforward network size of 1200, and 12 attention heads, totaling 14.5 million parameters.

### Layer Mapping and Strategy
- **Mapping Function**: `g(m) = 3 × m`, chosen to maximize coverage with fewer student layers, enabling effective capture of essential features despite the reduced scale.

### Distillation Objectives and Parameters
- **Temperature**: 1 (for direct probability transfer and maintaining fidelity to the teacher model’s outputs).
- **Layer Contribution Weighting**: Equal (λ = 1), simplifying the loss computation and focusing on balanced knowledge transfer.
- **Data Utilized**: Original training data plus unlabelled augmented data from GPT 3.5T and GPT 4o.

### Distillation Phases
- **Intermediate Layer Distillation**:
  - **Duration**: 20 epochs
  - **Batch Size**: 32
  - **Learning Rate**: 5e-5
  - **Max Sequence Length**: 64 tokens
  - **Warm-Up Proportion**: 0.1
  - **Focus**: Adapting student model’s intermediate representations to closely mirror those of the teacher, leveraging augmented data.

- **Prediction Layer Distillation**:
  - **Duration**: 3 epochs
  - **Other Parameters**: Same as Intermediate Layer Distillation
  - **Focus**: Refining the output layer’s decision-making process to replicate the teacher model.

### Comprehensive Approach
- **Two-Phase Distillation**: By dividing the distillation into two distinct phases, TinyFinBERT learns both robust deep-layer features and fine-tunes its output predictions, ensuring close alignment with the Augmented FinBERT and enhancing its performance in complex financial sentiment analysis tasks.

### Process Overview and Data Usage
- **Datasets Used**: Financial Phrasebank dataset alongside unlabelled data augmented by GPT-3.5T and GPT-4o
- **Teacher Model**: Augmented FinBERT
- **Student Model**: TinyFinBERT
  - **Transformer Layers**: 4
  - **Hidden Size**: 312
  - **Feedforward Network Size**: 1200
  - **Attention Heads**: 12
  - **Total Parameters**: 14.5 million

### Layer Mapping and Strategy






### Comprehensive Approach
By dividing the distillation into two distinct phases, TinyFinBERT not only learns robust features from the deeper layers of Augmented FinBERT but also fine-tunes its output predictions to closely align with those of the teacher model. This structured approach optimizes the transfer of nuanced understanding from Augmented FinBERT, enhancing TinyFinBERT’s performance in complex financial sentiment analysis tasks.


## Contribution of This Work
This work presents several novel contributions to the field of domain-specific language modeling:

1. **Structured Knowledge Distillation:** Introduce a two-tiered knowledge distillation approach using LLM-generated synthetic data. This methodology enables FinBERT to serve as an enriched teacher model and subsequently distills its knowledge into TinyFinBERT, yielding a compact, efficient model for financial sentiment analysis.

2. **Synthetic Data Generation with LLMs:** By employing both GPT-4 Omni and GPT-3.5 Turbo, we address data scarcity and boost the robustness of training datasets. This augmentation enhances FinBERT’s accuracy and helps generalize TinyFinBERT for real-world financial language processing tasks.

3. **Comprehensive Evaluation on Multiple Datasets:** TinyFinBERT’s performance was evaluated on three key datasets—Financial PhraseBank, FiQA 2018 Task1, and Forex News Annotated—demonstrating its ability to generalize and retain accuracy across different types of financial texts.

These contributions underscore the effectiveness of our approach in developing efficient, high-performing models tailored for the financial domain.

## Citation
The code in this repository is developed for my [Masters Thesis](https://arxiv.org/abs/2409.18999). Please cite it if you find the repository helpful.

```
@article{thomas2024enhancing,
  title={Enhancing TinyBERT for Financial Sentiment Analysis Using GPT-Augmented FinBERT Distillation},
  author={Thomas, Graison Jos},
  journal={arXiv preprint arXiv:2409.18999},
  year={2024}
}
```

## License
This project is licensed under the MIT License see [License](https://github.com/graison-thomas/TinyFinBERT/blob/main/LICENSE).
