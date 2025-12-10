# Performance Comparison: Scratch vs. Transfer Learning on V2 Dataset

This document provides a final comparison between the two models developed in this project, both trained on the refined V2 (cropped) dataset.

### Summary Table

| Metric               | Model #1-V2 (Scratch) | Model #2-V2 (Transfer) | Conclusion                                                              |
| :------------------- | :-------------------: | :--------------------: | :---------------------------------------------------------------------- |
| Overall Accuracy     | 63%                   | **74%**                | Transfer learning is clearly superior, though shy of the 80% target on this harder, cleaner problem. |
| 'Rock' F1-Score      | 0.67                  | **0.80**               | Transfer model is significantly more reliable for 'rock'.                |
| 'Scissors' F1-Score  | 0.56                  | **0.75**               | **Key Improvement:** Transfer learning largely solved the `scissors` problem. |
| 'Paper' F1-Score     | 0.56                  | **0.68**               | Both models struggled, but transfer learning was still better.           |
| Training Stability   | Unstable              | **Stable**             | Demonstrates the robustness of pre-trained features on clean data.      |

### Final Conclusion

This project successfully demonstrates a complete machine learning workflow, from baseline modeling to iterative data refinement. The initial experiments on the V1 (uncropped) data proved that even a powerful transfer learning model can produce misleadingly high accuracy by learning from spurious correlations. By engineering a human-in-the-loop cropping pipeline to create a clean V2 dataset, we forced the models to address the true problem.

The final **74% accuracy** of the transfer learning model on this clean dataset represents a more honest and valuable result than the 83% achieved on flawed data. It highlights the core principle of machine learning: **data quality is paramount**, and a slightly lower accuracy on the correct problem is infinitely more valuable than a high accuracy on the wrong one.