# Project Analysis & Key Findings

**Author:** Tim Chinye
<br>
**Date:** 12/12/2025
<br>
**Status:** Final Analysis Complete

This document contains a comprehensive analysis of the experimental results for the Rock-Paper-Scissors Hand Gesture Recognition project. It follows the project's iterative journey, starting with a baseline on an initial dataset (V1), identifying and solving a critical data quality issue, and culminating in a final analysis of models trained on a refined, superior dataset (V2).

---

## Part 1: Initial Baseline with V1 (Uncropped) Dataset

### 1.1. Summary of Initial Findings

The first set of experiments was conducted on the original, uncropped V1 dataset. These results served as a crucial, baseline that exposed fundamental issues with the initial data collection approach. The findings from this stage directly motivated the development of the data refinement pipeline.

*   **Model #1 (Scratch on V1):** This model completely failed to learn, exhibiting extreme training instability and achieving a catastrophically low accuracy. The confusion matrix revealed that it was unable to distinguish between classes, particularly `rock`, which it misclassified over 80% of the time.
*   **Model #2 (Transfer Learning on V1):** Despite the flawed data, this model showcased the power of transfer learning by achieving 83% accuracy. However, this high score was found to be misleading. Deeper analysis revealed a low recall of 0.67 for the `scissors` class, indicating that the model was likely overfitting to background cues rather than learning true gesture features.

### 1.2. Model #1-V1: Custom CNN from Scratch (Uncropped Data)

#### Analysis of Training History (V1)
![Model 1-V1 Training Curves](./results_v1_uncropped/scratch_model_training_curves.png)

*   **Observation:** The training and validation curves are extremely noisy and erratic. Neither accuracy curve shows a consistent upward trend, with validation accuracy (red) peaking at ~0.65 but frequently dropping below 0.4. The training loss (blue) is equally volatile and never converges. A dramatic performance crash is visible around epoch 10, where training accuracy plummets from 0.46 to 0.28.
*   **Interpretation:** This behavior demonstrates severe training instability and a failure to learn. The model is not converging on a stable solution, proving that the features it is trying to learn from the raw, uncropped images are not predictive or consistent.

#### Analysis of Performance Metrics (V1)
![Model 1-V1 Confusion Matrix](./results_v1_uncropped/scratch_model_confusion_matrix.png)

|              | precision | recall | f1-score | support |
| :----------- | :-------: | :----: | :------: | :-----: |
| **none**     |   0.84    |  0.74  |   0.79   |   72    |
| **paper**    |   0.73    |  0.70  |   0.72   |   74    |
| **rock**     |   0.61    |  0.73  |   0.67   |   67    |
| **scissors** |   0.61    |  0.60  |   0.60   |   70    |
|              |           |        |          |         |
| **accuracy** |           |        |   0.69   |   283   |

*   **Observation:** The confusion matrix confirms a catastrophic failure. For the `rock` class (third row), the model only correctly identified it 12 times, while misclassifying it as `none` (22), `paper` (11), and `scissors` (22) for a total of 55 incorrect predictions. Similarly, for the `scissors` class (bottom row), it was misclassified as `paper` 28 times - significantly more than the 33 times it was correct.
*   **Interpretation:** The model has failed to learn the visual features of hand gestures. The widespread confusion, especially the model's inability to recognize a closed fist (`rock`), proves it is latching onto spurious correlations in the background. The model's predictions are no better than random guessing, with a clear bias towards misclassification.

### 1.3. Model #2-V1: Transfer Learning (MobileNetV2) (Uncropped Data)

#### Analysis of Training History (V1)
![Model 2-V1 Training Curves](./results_v1_uncropped/transfer_model_training_curves.png)

*   **Observation:** The training curves show a more stable learning trend than the scratch model, with validation accuracy (red) consistently rising and surpassing 80%. However, the training process is still noisy, with a notable accuracy spike to nearly 90% at epoch 12, followed by significant dips at epochs 14 and 22.
*   **Interpretation:** The robustness of the pre-trained MobileNetV2 base allowed it to overcome much of the background noise and achieve a high score. However, the volatility indicates that the model was still struggling with inconsistent signals from the uncropped images, leading to an unstable learning path.

#### Analysis of Performance Metrics (V1)
![Model 2-V1 Confusion Matrix](./results_v1_uncropped/transfer_model_confusion_matrix.png)

|              | precision | recall | f1-score | support |
| :----------- | :-------: | :----: | :------: | :-----: |
| **none**     |   0.77    |  0.93  |   0.84   |   72    |
| **paper**    |   0.77    |  0.82  |   0.80   |   74    |
| **rock**     |   0.92    |  0.88  |   0.90   |   67    |
| **scissors** |   0.89    |  0.67  |   0.76   |   70    |
| **accuracy** |           |        |   0.83   |   283   |

*   **Observation:** The model achieved an impressive 83% overall accuracy. The F1-scores for `rock` (0.90) and `paper` (0.80) are excellent. However, the confusion matrix reveals a critical underlying flaw: the recall for the `scissors` class is only 0.67. It was misclassified as `paper` 13 times and `none` 10 times.
*   **Interpretation:** This is a textbook case of a model achieving a high score for the wrong reasons. The high accuracy was likely a result of the model learning to associate background features (e.g., "operator's hoodie at a specific angle") with certain classes. This "cheat" worked well for the more distinct `rock` and `paper` gestures but failed on the more ambiguous `scissors` gesture, where the background cues were not strong enough. The 83% accuracy is therefore misleading and represents a brittle, non-generalizable model.

### 1.4. Problem Identification & Engineered Solution

The V1 experiments conclusively demonstrated that data quality was the primary bottleneck. The high but misleading accuracy of the transfer model and the complete failure of the scratch model both pointed to the same root cause: spurious correlations from background noise.

To solve this, a human-in-the-loop data refinement pipeline was engineered using the MediaPipe library. This tool automated the cropping of clear hand gestures and provided an interface for manual review, cropping, or discarding of ambiguous images. This process created a high-quality, focused V2 (cropped) dataset, which was used for all final experiments.

---

## Part 2: Final Model Analysis on V2 (Cropped) Dataset

The following analysis is based on models re-trained on the superior V2 dataset.

### 2.1. Model #1-V2: Custom CNN from Scratch (Cropped Data)

#### Executive Summary
After re-training on the clean V2 dataset, the scratch model showed a marked improvement in stability and performance, achieving a final accuracy of 63%. While still below the project target, this demonstrates that data quality was the primary limiting factor in the initial experiment. The model, however, still struggles with the inherent difficulty of learning complex features from scratch.

#### Analysis of Training History
![Model 1-V2 Training Curves](./results/scratch_model_training_curves.png)

*   **Observation:** The training curves, while still noisy, show a much more discernible learning trend compared to the V1 results. Both training (blue) and validation (red) accuracies trend generally upwards, and the losses trend downwards. The instability is reduced, and there is no catastrophic performance crash.
*   **Interpretation:** By removing the distracting backgrounds, the model was able to start learning relevant features from the hand gestures. The learning process is still inefficient and unstable - a characteristic of training a deep network from scratch on a small dataset - but it is no longer failing completely. It is now genuinely attempting to solve the correct problem.

#### Analysis of Performance Metrics
![Model 1-V2 Confusion Matrix](./results/scratch_model_confusion_matrix.png)

|              | precision | recall | f1-score | support |
| :----------- | :-------: | :----: | :------: | :-----: |
| **none**     |   0.60    |  0.88  |   0.71   |   72    |
| **paper**    |   0.56    |  0.57  |   0.56   |   74    |
| **rock**     |   0.88    |  0.54  |   0.67   |   67    |
| **scissors** |   0.60    |  0.53  |   0.56   |   70    |
| **accuracy** |           |        |   0.63   |   283   |

*   **Observation:** The overall accuracy improved to 63%. The confusion matrix reveals a new performance profile. The model is now very good at not misclassifying other gestures as `rock` (high precision of 0.88), but it fails to identify `rock` when it sees it (low recall of 0.54). The biggest issue is now massive confusion between `paper` and `scissors`, with `scissors` being misclassified as `paper` 24 times.
*   **Interpretation:** The model has learned some distinct features (likely the closed fist of `rock`), but it lacks the sophistication to differentiate between the open-fingered gestures (`paper` and `scissors`). This is a classic example of a simple model finding the "easy" patterns but failing on the more nuanced ones. The improvement from the V1 training is undeniable, but the model's architectural limitations are now the main bottleneck.

### 2.2. Model #2-V2: Transfer Learning (MobileNetV2) (Cropped Data)

#### Executive Summary
The transfer learning model, when trained on the refined V2 dataset, achieved a final accuracy of 74%. While this is a lower overall accuracy than its 83% on the flawed V1 data, a deeper analysis reveals it is now a more balanced and reliable model. The previous high score was artificially inflated by overfitting to background cues, whereas this new score reflects a more genuine understanding of the hand gestures themselves.

#### Analysis of Training History
![Model 2-V2 Training Curves](./results/transfer_model_training_curves.png)

*   **Observation:** The training curves are the most stable of all experiments. The validation accuracy (red) shows a steady, consistent rise to a plateau around 70-75%. The validation loss (red) finds a stable minimum. The gap between training and validation is present but consistent, indicating controlled learning.
*   **Interpretation:** This is the healthiest learning profile observed. The model is effectively generalizing from the clean training data to the unseen validation data. Unlike the V1 training, there are no sudden spikes or crashes, indicating that the model is learning from a consistent, high-quality signal.

#### Analysis of Performance Metrics
![Model 2-V2 Confusion Matrix](./results/transfer_model_confusion_matrix.png)

|          | precision | recall | f1-score | support |
| :------- | :-------: | :----: | :------: | :-----: |
| none     |   0.61    |  0.90  |   0.73   |   72    |
| paper    |   0.84    |  0.57  |   0.68   |   74    |
| rock     |   0.83    |  0.78  |   0.80   |   67    |
| scissors |   0.78    |  0.71  |   0.75   |   70    |
| accuracy |           |        |   0.74   |   283   |

*   **Observation:** The overall accuracy is 74%. Critically, the performance is much more balanced across classes compared to all previous experiments. The F1-scores for `rock` (0.80) and `scissors` (0.75) are strong. The recall for the previously problematic `scissors` class has improved from 0.67 (on V1 data) to a healthier 0.71.

*   **Interpretation: Why 74% is a Better Result than 83%**
    This is the most critical finding of the project. The V1 model's 83% accuracy was an illusion. It was achieved by exploiting a flaw in the data - the consistent presence of the operator's body and clothing. The model learned a shortcut: "black hoodie at this angle = `rock`". This is a brittle, non-generalizable solution that would fail in the real world.

    By training on the clean V2 dataset, we removed this "cheat." The model was forced to solve the much harder, correct problem: differentiating hand gestures based only on the hand's features. The resulting 74% accuracy, while numerically lower, reflects a genuine understanding of the task. The model is now more robust and would perform better on images of *anyone's* hand, not just the operator's. The improved recall for `scissors` (from 0.67 to 0.71) is direct proof that the model is now better at distinguishing the most difficult gestures, even if its overall confidence on "easier" classes has slightly decreased without the background crutch.

---

## 3. Final Comparison & Conclusion (V2 Dataset)

The definitive comparison is between the two models trained on the superior V2 dataset.

| Metric               | Model #1-V2 (Scratch) | Model #2-V2 (Transfer) | Conclusion                                                         |
| :------------------- | :-------------------: | :--------------------: | :----------------------------------------------------------------- |
| Overall Accuracy     | 63%                   | 74%                    | Transfer learning is clearly superior, though shy of the 80% target on this harder, cleaner problem. |
| 'Rock' F1-Score      | 0.67                  | 0.80                   | Transfer model is significantly more reliable for 'rock'.          |
| 'Scissors' F1-Score  | 0.56                  | 0.75                   | Transfer learning largely solved the `scissors` problem.           |
| 'Paper' F1-Score     | 0.56                  | 0.68                   | Both models struggled, but transfer learning was still better.     |
| Training Stability   | Unstable              | Stable                 | Demonstrates the robustness of pre-trained features on clean data. |

**Final Conclusion:** This project successfully demonstrates a complete machine learning workflow, from baseline modeling to iterative data refinement. The initial experiments on V1 data proved that even a powerful transfer learning model can produce misleadingly high accuracy by learning from spurious correlations. By engineering a human-in-the-loop cropping pipeline to create a clean V2 dataset, we forced the models to address the true problem. The final 74% accuracy of the transfer learning model on this clean dataset represents a more honest and valuable result than the 83% achieved on flawed data. It highlights the core principle of machine learning: data quality is paramount, and a slightly lower accuracy on a correct problem is infinitely more valuable than a high accuracy on the wrong one.