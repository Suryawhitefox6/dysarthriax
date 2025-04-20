# Dysarthix:Dysarthric Speech Detection and Severity Classification
---

## Abstract

Automatic dysarthric speech detection is a crucial task for identifying and classifying dysarthria, a motor speech disorder characterized by poor articulation resulting from neurological disorders. This project explores three neural network architectures: CNN-GRU (Convolutional Neural Network with Gated Recurrent Units), CNN-BiLSTM (Convolutional Neural Network with Bidirectional Long Short-Term Memory), and a transfer learning method utilizing Wav2Vec2. Our experimental results demonstrate the strong effectiveness of these methods, with the CNN-GRU model achieving 96.51% accuracy, the CNN-BiLSTM model at 94.36% accuracy, and the Wav2Vec2 model at 93.85% accuracy. The CNN-GRU structure performed best by efficiently representing local acoustic features and temporal relations in speech signals. This work lays the foundation for developing clinical tools for early detection and treatment of dysarthria and refining automatic speech recognition systems for patients with dysarthria.

**Keywords:** Dysarthria Detection, Speech Analysis, CNN-GRU, CNN-BiLSTM, Wav2Vec2, Acoustic Features, Deep Learning, TORGO Dataset

---

## Table of Contents

1. [Introduction](#introduction)
2. [Literature Survey](#literature-survey)
3. [Proposed Methodology](#proposed-methodology)
    - 3.1 [Overview of Proposed Work](#overview-of-proposed-work)
    - 3.2 [Dataset Description](#dataset-description)
    - 3.3 [Models Used](#models-used)
        - 3.3.1 [CNN-GRU Model](#cnn-gru-model)
        - 3.3.2 [CNN-BiLSTM Model](#cnn-bilstm-model)
        - 3.3.3 [Wav2Vec Model](#wav2vec-model)
4. [Results and Analysis](#results-and-analysis)
    - 4.1 [Results of CNN-GRU Model](#results-of-cnn-gru-model)
    - 4.2 [Results of CNN-BiLSTM Model](#results-of-cnn-bilstm-model)
    - 4.3 [Results of Wav2Vec Model](#results-of-wav2vec-model)
5. [Conclusion and Future Work](#conclusion-and-future-work)
    - 5.1 [Research Conclusion](#research-conclusion)
    - 5.2 [Discussion](#discussion)
    - 5.3 [Future Scope](#future-scope)
6. [References](#references)

---
---
---

## List of Abbreviations

- **CNN**: Convolutional Neural Network
- **GRU**: Gated Recurrent Unit
- **BiLSTM**: Bidirectional Long Short-Term Memory
- **MFCC**: Mel-Frequency Cepstral Coefficients
- **TORGO**: Toronto Rehabilitation Institute Gerontology (dataset name)
- **ASR**: Automatic Speech Recognition
- **Wav2Vec2**: A self-supervised learning framework for speech recognition
- **ReLU**: Rectified Linear Unit
- **LSTM**: Long Short-Term Memory
- **ROC**: Receiver Operating Characteristic
- **AUC**: Area Under the Curve

---

## Chapter 1: Introduction

Dysarthria is a speech disorder that disrupts the ability of an individual to produce understandable speech. It is a serious neurological defect that can affect people of virtually any age and background, typically resulting from disorders like cerebral palsy, stroke, traumatic brain injury, and multiple sclerosis. Dysarthria detection involves identifying dysarthric speech versus non-dysarthric speech, and severity grading refers to grading the condition into various levels of severity, i.e., mild, moderate, or severe. This classification is beneficial as it enables clinicians to offer specific treatment of speech therapy for a patient depending on the severity of impairment of speech.

Recent advances in deep learning have made it possible to build automated systems for the detection of dysarthric speech. This project investigates three deep learning architectures: CNN-GRU, CNN-BiLSTM, and Wav2Vec2. These models are evaluated using the TORGO dataset, which consists of speech recordings of control and dysarthric speakers. The performance of each model is assessed using classification metrics such as accuracy, precision, recall, and F1-score, as well as speech-specific metrics like Word Error Rate (WER) and Phoneme Error Rate (PER).

---

## Chapter 2: Literature Survey

This chapter reviews various studies and methodologies proposed for dysarthric speech detection and severity classification. Key contributions include:

- **Irshad et al. (2024)**: Proposed UTrans-DSR, a novel transformer architecture for dysarthric speech recognition, achieving 97.75% accuracy.
- **Vinotha R et al. (2024)**: Introduced a model integrating OpenAI's Whisper model for better speech recognition in dysarthria patients, achieving a mean Word Recognition Accuracy (WRA) of 74.08%.
- **Mahendran M et al. (2023)**: Suggested a CNN-based method for dysarthria detection, achieving 93.87% accuracy.
- **Joshy and Rajeev Rajan (2023)**: Proposed a multi-head attention (MHA) and multi-task learning (MTL) mechanism for classifying dysarthria severity, achieving 95.75% accuracy.
- **Latha M et al. (2023)**: Proposed a hybrid deep learning-based model for dysarthric speech synthesis and recognition, achieving 96% accuracy.
- **Al-Ali et al. (2025)**: Suggested a two-stage method for dysarthria severity level classification using binary classification with clustering methods.
- **Salim et al. (2024)**: Proposed a multi-modal dysarthric speaker verification system integrating data augmentation with feature fusion methods.
- **Sajiha et al. (2024)**: Put forward a CWT-layered CNN model for automatic dysarthria detection and severity level assessment.
- **Lin et al. (2023)**: Proposed an automatic speech recognition (ASR) system for speech disorder users, addressing data scarcity and articulatory impairments.
- **Javanmardi et al. (2023)**: Researched automatic dysarthric speech classification and severity estimation using wav2vec 2.0.

---

## Chapter 3: Proposed Methodology

### 3.1 Overview of Proposed Work

The proposed pipeline takes raw `.wav` audio files and classifies speech as normal or dysarthric, then evaluates its severity. The pipeline includes input processing, feature extraction, and classification stages.

### 3.2 Dataset Description

The TORGO dataset, totaling 1.75GB, includes 17,633 audio samples from dysarthric and non-dysarthric speakers. The dataset is balanced and categorized by gender and speech condition.

#### 3.2.1 Data Composition

| Speakers     | Control (Non-Dysarthric) | Dysarthric | Total   |
|--------------|--------------------------|------------|---------|
| Male         | 6,778                    | 3,787      | 10,565  |
| Female       | 4,677                    | 2,391      | 7,068   |
| **Total**    | **11,455**               | **6,178**  | **17,633** |

#### 3.2.2 Recording Structure and Organization

The dataset is structured into four primary directories:
- **F_Con**: Female Control
- **F_Dys**: Female Dysarthric
- **M_Con**: Male Control
- **M_Dys**: Male Dysarthric

Each directory contains subfolders organized by individual speakers and recording sessions.

#### 3.2.3 Session and Recording Information

Each speaker's folder includes several recording sessions captured with array and headset microphones. The dataset provides varied acoustic conditions and is balanced, making it suitable for machine learning-based tasks related to speech.

### 3.3 Data Preprocessing

Raw audio samples are preprocessed to derive numerical features based on Mel Frequency Cepstral Coefficients (MFCCs). Each audio signal is mapped to a 128 MFCC feature vector and standardized to a fixed maximum length of 3 seconds, sampled at 16 kHz. The dataset is split into 80% training and 20% validation sets.

### 3.4 Models Implemented

#### 3.4.1 CNN-GRU Model

The CNN-GRU model combines Convolutional Neural Networks (CNN) for feature extraction and Gated Recurrent Units (GRU) for capturing temporal dependencies in speech signals. The model architecture includes convolutional layers, GRU layers, and fully connected layers for classification.

#### 3.4.2 CNN-BiLSTM Model

The CNN-BiLSTM model builds upon CNN-based feature extraction with the addition of Bidirectional Long Short-Term Memory (BiLSTM) layers. The BiLSTM layers process speech signals both forward and backward, enhancing the model's ability to classify dysarthric and non-dysarthric speech.

#### 3.4.3 Wav2Vec2 Model

The Wav2Vec2 model is a self-supervised learning model that obtains sophisticated speech representations directly from raw audio inputs. It is pre-trained on vast speech databases and fine-tuned on dysarthric speech data for accurate detection and estimation of dysarthria severity.

---

## Chapter 4: Results and Analysis

### 4.1 Performance Metrics

The performance of the proposed models is evaluated using the following metrics:
- **Accuracy**: Overall performance of the model.
- **Loss**: Categorical cross-entropy loss function.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to detect all instances of dysarthric speech.
- **F1 Score**: Harmonic mean of precision and recall.

### 4.2 Results of CNN-GRU Model

The CNN-GRU model achieved 96.51% accuracy, demonstrating its exceptional generalizability on unattended speech samples. The model's training loss of 0.1110 and validation loss of 0.1226 indicate significantly reduced classification errors.

### 4.3 Results of CNN-BiLSTM Model

The CNN-BiLSTM model achieved 94.36% accuracy, with strong binary classification capability. The model shows a slight bias toward control classification, with higher false negatives than false positives.

### 4.4 Results of Wav2Vec2 Model

The Wav2Vec2 model achieved 93.85% validation accuracy, indicating strong generalization to unseen data. The model's training loss of 0.3739 and training accuracy of 92.08% suggest effective learning and reliable detection of dysarthric speech.

### 4.5 Comparative Analysis

| Model       | Accuracy | Loss   | Type       | Precision | Recall | F1 Score |
|-------------|----------|--------|------------|-----------|--------|----------|
| CNN-GRU     | 96.51%   | 0.1226 | Control    | 0.98      | 0.97   | 0.97     |
|             |          |        | Dysarthric | 0.94      | 0.96   | 0.95     |
| CNN-BiLSTM  | 94.36%   | 0.7239 | Control    | 0.94      | 0.98   | 0.96     |
|             |          |        | Dysarthric | 0.96      | 0.92   | 0.91     |
| Wav2Vec2    | 93.85%   | 0.3739 | Control    | 0.96      | 0.95   | 0.93     |
|             |          |        | Dysarthric | 0.92      | 0.90   | 0.95     |

---

## Chapter 5: Conclusion and Future Work

### 5.1 Research Conclusion

The CNN-GRU, CNN-BiLSTM, and Wav2Vec2 models demonstrated strong potential for the early diagnosis of dysarthria, with accuracy scores of 96.51%, 94.36%, and 93.85%, respectively. These models can aid clinical diagnosis and intervention planning.

### 5.2 Discussion

The models performed well in dysarthric speech detection but faced challenges in severity classification. Future work could include integrating prosodic or visual characteristics and data augmentation processes to improve performance.

### 5.3 Future Scope

Future enhancements could involve real-time use with immediate feedback loops, multimodal approaches, and exploring techniques like transfer learning to increase functionality in research and clinical settings.

---

## Chapter 6: References

1. U. Irshad et al., “UTran-DSR: a novel transformer-based model using feature enhancement for dysarthric speech recognition,” EURASIP Journal on Audio Speech and Music Processing, vol. 2024, no. 1, Oct. 2024.
2. R. Vinotha, D. Hepsiba, and L. D. Vijay Anand, “Leveraging OpenAI Whisper Model to Improve Speech Recognition for Dysarthric Individuals,” pp. 1–5, Jul. 2024.
3. M. Mahendran, R. Visalakshi, and S. Balaji, “Dysarthia detection using convolution neural network,” Oct. 2023.
4. A. A. Joshy and R. Rajan, “Dysarthria severity classification using multi-head attention and multi-task learning,” Speech Communication, vol. 147, pp. 1–11, Dec. 2022.
5. M. Latha, M. Shivakumar, G. R. Manjula, M. A. Hemakumar, and M. K. Kumar, “Deep Learning-Based Acoustic Feature Representations for Dysarthric Speech Recognition,” SN computer science, vol. 4, no. 3, pp. 1–7, Mar. 2023.
6. A. Al-Ali, R. M. Haris, Y. Akbari, M. Saleh, S. Al-Máadeed, and R. Kumar M, “Integrating binary classification and clustering for multi-class dysarthria severity level classification: a two-stage approach,” Cluster Computing, vol. 28, no. 2, Nov. 2024.
7. S. Salim, S. Shahnawazuddin, W. Ahmad. "Combined approach to dysarthric speaker verification using data augmentation and feature fusion," in Speech Communication, vol. 160, pp. 103070, 2024.
8. Sajiha, S., et al. "Automatic dysarthria detection and severity level assessment using CWT-layered CNN model," in EURASIP Journal on Audio, Speech, and Music Processing, vol. 2024, no. 1, pp. 33, 2024.
9. Lin, Y., et al. "Disordered speech recognition considering low resources and abnormal articulation," in Speech Communication, vol. 155, pp. 103002, 2023.
10. Javanmardi, F., et al, "Wav2vec-based detection and severity level classification of dysarthria from speech," in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

---

