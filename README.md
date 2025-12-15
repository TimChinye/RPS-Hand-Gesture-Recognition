# Rock-Paper-Scissors Hand Gesture Recognition System

This project implements a complete, end-to-end deep learning system for recognizing 'Rock', 'Paper', and 'Scissors' hand gestures in real-time. It's not just a classifier, but a full machine learning application, encompassing a custom data collection tool, a sophisticated data processing pipeline, model training/evaluation, and an interactive game with an intelligent AI opponent.

The development followed an iterative process. Initial experiments revealed that background noise in the raw dataset was a critical bottleneck, leading to models that achieved high accuracy for the wrong reasons. To solve this, a human-in-the-loop data cropping and review pipeline was engineered using MediaPipe. This ensured the final models were trained on high-quality, focused images of hand gestures, resulting in a robust and genuinely accurate system.

The final deliverable is an interactive game where the user plays against a "smart" AI opponent that uses an LSTM (Long Short-Term Memory) network to learn the player's move history and predict their next action, making for a challenging and dynamic experience.

## Features

-   **Interactive Data Collection:** A Python script (`src/data_collection.py`) using OpenCV to capture images directly from a webcam, with on-screen controls for switching classes and managing the capture process.
-   **Human-in-the-Loop Data Processing Pipeline:**
    -   **Auto-Cropping:** Uses the MediaPipe library to automatically detect and crop hands from the raw images, eliminating background noise.
    -   **Manual Review System:** An interactive OpenCV-based tool for reviewing images where automatic cropping failed, allowing for manual cropping, rejection, or replacement.
    -   **Dataset Assembly:** Scripts to build the final, clean dataset and split it reproducibly into training, validation, and test sets.
-   **Dual Model Architectures:**
    1.  **Scratch CNN:** A custom-built Convolutional Neural Network to serve as a baseline.
    2.  **Transfer Learning:** A fine-tuned MobileNetV2 model that leverages pre-trained weights for superior performance.
-   **Comprehensive Training & Evaluation:** A unified training script (`src/train.py`) that trains both models, saves the results, and generates performance reports, including:
    -   Training/Validation history plots (Accuracy & Loss).
    -   Classification reports with precision, recall, and F1-scores.
    -   Confusion matrices for detailed error analysis.
-   **Advanced Interactive Game:**
    -   **Real-time Gesture Recognition:** Uses the trained model to classify the player's hand gesture live via webcam.
    -   **Predictive AI Opponent (LSTM):** Instead of random moves, the AI uses an LSTM network to analyze the player's move history and predict their next gesture, then selects the counter-move.
    -   **System Calibration:** An initial calibration phase to learn the "empty" scene and the "operator present" scene, preventing false positives when no gesture is being made.
    -   **Model Explainability (Grad-CAM):** An "Analysis Mode" that visualizes the model's decision-making process using Grad-CAM heatmaps, showing which parts of the image were most influential.

## Final Model Performance

The final system uses the **Transfer Learning (MobileNetV2) model trained on the V2 (cropped) dataset**. This model successfully meets the project's target of >80% accuracy and demonstrates robust performance across all classes, proving it has learned the true features of the hand gestures, not background noise.

| Metric            | Model #2-V2 (Transfer Learning) |
| :---------------- | :-----------------------------: |
| **Overall Accuracy**  | **82%**                         |
| 'Rock' Recall     | 0.95                            |
| 'Scissors' Recall | 0.89                            |
| 'Paper' Recall    | 0.61                            |

A detailed breakdown and comparative analysis of all model experiments can be found in `analysis.md`.

## Option A: Running a live, interactive (Google Colab - Recommended)

### Project Links

*   **GitHub Repository:** https://github.com/TimChinye/RPS-Hand-Gesture-Recognition
*   **Google Colab Notebook:** https://colab.research.google.com/drive/1gwBIG6BgqVgVZ7dlufnEqwTbDG_rxGmk?usp=sharing
*   **Dataset & Trained Models (Google Drive):** https://drive.google.com/drive/folders/1Xlbh93dTr5OEQoiqRSaBLWkjedBfQxNy?usp=sharing

The Google Colab notebook is a live, interactive report that demonstrates the entire ML pipeline, from data preparation to live model evaluation.

1.  Open the **Google Colab Notebook** linked above.
2.  Click **"Runtime" -> "Run all"**.
3.  The notebook will automatically clone the GitHub repository, download the dataset and pre-trained models from Google Drive, and run the complete evaluation, displaying all results inline.

## Option B: Running the Full Project Locally

This option allows you to run the entire data pipeline, train the models from scratch, and play the interactive game on your local machine.

### Installation

This project uses Conda to manage its environment and dependencies (TensorFlow, OpenCV, et).

1.  **Prerequisites:** Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/TimChinye/RPS-Hand-Gesture-Recognition/.git
    cd C://path/to/RPS-Hand-Gesture-Recognition
    ```

3.  **Create and activate the Conda environment:**
    ```bash
    # It is recommended to create an environment from the provided yml file
    conda env create -f environment.yml
    ```

    After creating with the previous command, activate it with the following command:
    ```bash
    # Before copying, replace "%USERPROFILE%" with whatever's most appropiate.
    %USERPROFILE%/miniconda3/Scripts/activate
    conda activate .\.conda
    ```

### Usage: The Full ML Pipeline

The entire project workflow is managed through `run.py`. Depending on your goal, you can choose one of the following paths.

#### Path 1: Play Immediately (Quick Start)
If you just want to run the game using pre-trained models, you can skip the data collection and training steps.

1.  Download the "Dataset" and "Trained Models" from the [Google Drive folder](https://drive.google.com/drive/folders/1Xlbh93dTr5OEQoiqRSaBLWkjedBfQxNy?usp=sharing).
2.  Extract `dataset_final.zip` into the project's root folder.
3.  Create a `saved_models` folder in the root and move the two `.keras` files into it.
4.  You can now skip to **Step 4** and run the game directly:
    ```bash
    python run.py play
    ```

#### Path 2: Train on Provided Data
If you want to train the models yourself but skip the manual data collection, follow these steps.

1.  Download the "Dataset" from the Google Drive folder and extract it.
2.  You can now skip to **Step 2, part 4 (`prepare`)** and then proceed with training and playing.

#### Path 3: Run the Full Pipeline from Scratch
To collect your own data and build everything from the ground up, follow all the detailed steps below.

##### Step 0: Run it all at once

To run all steps from data collection to training in sequence, execute the following commands. Each step is explained in detail in the subsequent steps.

```bash
python run.py collect
python run.py crop
python run.py review
python run.py build
python run.py prepare
python run.py train
python run.py play
echo All RPS Workflow steps has been executed. 
```

##### Step 1: Collect Raw Image Data

Open the interactive data collection tool. Use the on-screen keys to switch between classes (`r`, `p`, `s`, `n`) and press `c` to start a capture countdown. These will be used to train, test, and validate. For the best results, use various angles, lighting settings, and positions.

```bash
python run.py collect
```

##### Step 2: Process and Prepare the Dataset

This multi-stage pipeline cleans the raw data and prepares it for training.

1.  **Auto-Crop:** Automatically detects and crops hands from the raw `dataset/` folder, saving them to `dataset_cropped/`.
    ```bash
    python run.py crop
    ```

2.  **Review (Optional but Recommended):** Manually review images that could not be auto-cropped and decide to keep, discard, or manually crop them.
    ```bash
    python run.py review
    ```

3.  **Build Final Dataset:** Combines the successfully cropped and reviewed images into `dataset_final/`.
    ```bash
    python run.py build
    ```

4.  **Prepare for Training:** Splits the final dataset into `train`, `validation`, and `test` sets within the `data/` directory.
    ```bash
    python run.py prepare
    ```

##### Step 3: Train the Models

This command will train both the scratch CNN and the transfer learning model on the data prepared in the previous step. The final models will be saved in `saved_models/` and all performance reports will be saved in `results/`.

```bash
python run.py train
```
*Note: To compare your results with the results referenced in `analysis.md`, you can find the results in the versioned subfolders (e.g; `results/v2_cropped/`).*

##### Step 4: Play the Game!

Launch the interactive game. It will automatically load the best-performing model (`saved_models/transfer_model.keras`) and pit you against the smart LSTM-powered AI.

*Note: To replicate how the game would work using the models referenced in `analysis.md`, you will want to take the files from the versioned subfolders (e.g; `saved_models/v2_cropped/`) and move them into the parent `saved_models` folder.*

```bash
python run.py play
```

### Project Structure

```
/
├── data/                    # (Generated) Split data (train/val/test) for training.
├── dataset/                 # (User-Generated) Raw, uncropped images from data collection.
├── dataset_cropped/         # (Generated) Auto-cropped images from the pipeline.
├── dataset_final/           # (Generated) Final curated dataset ready for splitting.
├── dataset_review/          # (Generated) Collection of manually reviewed and sorted images.
├── notebooks/               # Jupyter notebooks for exploration and development.
├── results/                 # Saved model performance metrics (reports, matrices, plots).
├── saved_models/            # Saved final trained models (.keras files).
├── src/                     # All Python source code for the project.
│   ├── assets/              # UI images for the game.
│   ├── models/              # Model architecture definitions.
│   ├── utils/               # Helper scripts for the data processing pipeline.
│   ├── data_collection.py   # Script to collect image data.
│   ├── game.py              # The interactive game application.
│   └── train.py             # Script for training and evaluating models.
├── .gitignore               # Specifies files for Git to ignore.
├── analysis.md              # In-depth analysis of model performance.
├── environment.yml          # Avoids dependency errors and one-command setups.
├── README.md                # This file.
└── run.py                   # Main entry point to run all project commands.
```

---

## AI Transparency Statement

**AITS Descriptor:** AITS 2: AI for Shaping

In accordance with the assessment brief (Section 10), I have used Artificial Intelligence at the permitted level of AITS 2. The university's AITS table defines this level with two key statements:

1.  **Permitted AI Contribution:** AI can be used for "shaping parts of the activity. This includes **initial outlining, concept development, prompting thinking, and/or improving structure/quality of the final output.**"
2.  **Required Human Contribution:** "Most of the activity is human developed/generated. **AI ideas and suggestions are refined and reviewed.**"

I pledge that my use of AI adheres strictly to this framework. The following log provides irrefutable evidence of this process, demonstrating how each interaction was a permitted "shaping" activity and how every AI suggestion was critically "refined and reviewed" by me to meet the specific demands of the assessment brief.

The unabridged chat histories are provided in the `/AI-Prompts` folder.

## AI Prompt Log: Evidence of AITS-2 Compliance

---

### **1. Initial Outlining & Project Structuring**

> **Brief Requirement (Section 1):** To create a "**fully functional neural network system**... ensuring that your approach is both theoretically sound and practically applicable."
> <br>
> **Brief Requirement (Section 4.1):** To ensure that "**code is well-structured... for readability and reproducibility.**"

*   <details>
    <summary><strong>Prompt 1.1: Project Planning</strong></summary>

    *   **AITS-2 Activity:** `Initial Outlining`
    *   **My Prompt:** `"I have two complex assignment briefs... [insert briefs] Can you help me create a detailed, day-by-day project plan and a master checklist of all deliverables..."`
    *   **Outcome & Justification:** The AI-generated outline was a starting point. **I refined and reviewed** this plan, adapting the schedule to my own workflow and used the checklist to rigorously track my progress toward building the "fully functional system" required.
    </details>
    <!-- Personal reference: https://aistudio.google.com/prompts/12BVdQD4ovVUd29QRjXosqcczrFUIyOIY -->

*   <details>
    <summary><strong>Prompt 1.2: Code Structure</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality`
    *   **My Prompt:** `"What should the project structure look like... for a Python-based machine learning project that uses both Jupyter notebooks for experimentation and .py scripts for the final application."`
    *   **Outcome & Justification:** The AI suggested a professional template. **I refined and reviewed** this structure, adapting it to the specific needs of this project to create a final submission that is "well-structured", "reproducible" as demanded by the brief, and suitable for my environment and tech stack.
    <!-- Personal reference: https://aistudio.google.com/prompts/12BVdQD4ovVUd29QRjXosqcczrFUIyOIY -->
    </details>

---

### **2. Concept Development for Informed Decisions**

> **Brief Requirement (Learning Outcome 1):** To "**Evaluate and apply... relevant theories, concepts, principles, and practises**..."

*   <details>
    <summary><strong>Prompt 2.1: Environment Management Concept</strong></summary>

    *   **AITS-2 Activity:** `Concept Development`
    *   **My Prompt:** `"Venv vs Conda? What's the difference and which is more appropriate for a project with complex dependencies like TensorFlow and OpenCV on Windows?"`
    *   **Outcome & Justification:** The AI's explanation of these core concepts was critical. **I refined and reviewed** its advice, confirming it against external documentation, which led to my informed decision to use Conda. This was essential for correctly applying the "theories and principles" of a reproducible ML environment.
    <!-- Personal reference: https://aistudio.google.com/prompts/1vRWsQh78Q8AQ7ZgBluL1um6uLYKytlg4 -->
    </details>

*   <details>
    <summary><strong>Prompt 2.2: Data Augmentation Concept</strong></summary>

    *   **AITS-2 Activity:** `Concept Development`
    *   **My Prompt:** `"In the context of Keras's ImageDataGenerator, is data augmentation applied on-the-fly during training or used to create a larger static dataset beforehand?"`
    *   **Outcome & Justification:** The AI's clear explanation was a key piece of "concept development." **I refined and reviewed** this information, which directly informed my implementation of the training pipeline, a core "practice" for preventing overfitting in neural networks.
    <!-- Personal reference: https://aistudio.google.com/prompts/1rQB8PB8-zNO_KHYWucoyZOMCN6FKd5eP -->
    </details>

*   <details>
    <summary><strong>Prompt 2.3: Defining Dataset Boundaries & Edge Cases</strong></summary>

    *   **AITS-2 Activity:** `Concept Development` / `Prompting Thinking`
    *   **My Prompt (Iterative Conversation):** `"For clarity, when I'm taking pictures of my hand and random objects, should it also pick my hand inside a glove as r/p/s or 'none'?"` followed by `"What about a fingerless glove?"`
    *   **Outcome & Justification:** This interaction was a crucial "concept development" step for defining the scope and boundaries of my dataset. The AI's explanation of how a CNN learns *features* (like skin texture) rather than abstract *concepts* (like a "hand") prompted my thinking on how to handle edge cases. **I reviewed and refined** this advice, leading to a strategic decision:
        1.  My positive classes (`rock`, `paper`, `scissors`) would be strictly limited to my bare hand to create a well-defined and solvable problem.
        2.  "Hard negative" examples, such as hands with full or fingerless gloves, would be intentionally added to the `none` class to improve the model's robustness.
    *   This informed decision was critical for applying the "concepts and principles" of building a high-quality dataset and a robust classifier, directly addressing the module's learning outcomes.
    <!-- Personal reference: https://aistudio.google.com/app/prompts/1o38kUHxwNLqDyrGmN1Yat39jhEDyczPV -->
    </details>

---

### **3. Technical Troubleshooting for a Working Deliverable**

> **Brief Requirement (Section 4.1):** To ensure that "**Your code must be error free and ready to run.**"

*   <details>
    <summary><strong>Prompt 3.1: Environment Troubleshooting</strong></summary>

    *   **AITS-2 Activity:** `Prompting Thinking`
    *   **My Prompt:** `"CondaToSNonInteractiveError: Terms of Service have not been accepted..."`
    *   **Outcome & Justification:** The AI's response prompted me to shift my thinking from "this is a bug" to "this is an administrative step." **I reviewed** this suggestion, understood the context, and executed the required commands myself, which was a necessary step to produce an "error free and ready to run" deliverable.
    <!-- Personal reference: https://aistudio.google.com/prompts/1vRWsQh78Q8AQ7ZgBluL1um6uLYKytlg4 -->
    </details>

*   <details>
    <summary><strong>Prompt 3.2: Resolving a Critical `LinkError`</strong></summary>

    *   **AITS-2 Activity:** `Prompting Thinking` / `Concept Development`
    *   **My Prompt (Implicit through error logs):** My prompt was the direct output from my terminal: `"conda install -c conda-forge opencv ... LinkError: post-link script failed ... 'C:\Users\timch\Documents\University\AI-' is not recognized as an internal or external command"`. I also provided clarifying context like `"Underscore failed in the past, that's why I changed to dashes."` and `"The issue was with the '&', I just changed it to a 'n'."`
    *   **Outcome & Justification:** This interaction was a classic example of using AI for `prompting thinking`. The AI's initial advice was based on common best practices (e.g; using underscores). My critical feedback (`"Underscore failed..."`) forced a deeper analysis. The AI's subsequent explanation of *why* the `&` character was causing the `LinkError` was a key moment of `concept development`. **I refined and reviewed** this information, which confirmed my own diagnosis. This allowed me to confidently implement the solution (renaming the folder), which was essential for creating the required "**error free and ready to run**" code.
    <!-- Personal reference: https://aistudio.google.com/prompts/1o38kUHxwNLqDyrGmN1Yat39jhEDyczPV -->
    </details>

---

### **4. Code Generation for Boilerplate Tasks**

> **Brief Requirement (Section 3.1):** To "**write generic python code to capture several images per hand gestures category... through computer’s camera.**"
> <br>
> **Brief Requirement (LO3):** To "**implement... artificial neural networks to solve real-world artificial intelligent problems.**"

*   <details>
    <summary><strong>Prompt 4.1: Generating the Data Collection Script</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt:** `"Write a Python script using OpenCV to capture images from your webcam. The script should save images to structured folders (/dataset/rock, /dataset/paper, etc). Here's my existing project structure: [pasted project tree structure]"`
    *   **Outcome & Justification:** In line with AITS-2, I used AI to generate a boilerplate script for a standard, repetitive task. This improved the quality and speed of my initial setup. After generating three versions, **I refined and reviewed** them extensively, combined specific ideas and refactored it to make it my own.
    *   The AI's role was limited to shaping the initial tool; the core intellectual work of creating the dataset, a key part of solving this "real-world problem," was entirely my own.
    <!-- Personal reference: https://aistudio.google.com/prompts/1o38kUHxwNLqDyrGmN1Yat39jhEDyczPV -->
    </details>

*   <details>
    <summary><strong>Prompt 4.2: Refining the Final Submission Notebook</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt:** The conversation began with a strategic question: `"Based on the point I'm at now, what do I do to match that gold standard?"` I then provided my existing work (code, assets, draft notebooks) and asked the AI to help assemble the final, polished Google Colab notebook. This was followed by my own critical feedback, such as:
        *   Identifying a runtime bug: `"TypeError: run_split() got an unexpected keyword argument 'source_dir'"`
        *   Demanding better quality explanations: `"...there's no actual explanations going on. Like why did we choose 'that' batch size..."`
    *   **Outcome & Justification:** This interaction perfectly demonstrates the AITS-2 framework. The AI's role was for **improving structure/quality of the final output** by refactoring my existing, human-written code into the professional, reproducible notebook format we had discussed. The core intellectual work—the code, the models, the dataset—was entirely my own. Crucially, **I refined and reviewed** the AI's output, acting as the human developer directing the process. I identified a functional bug and demanded a higher standard of documentation, which the AI then incorporated. The final deliverable was a direct result of this iterative, human-led refinement process.
    <!-- Personal reference: https://aistudio.google.com/prompts/1xWWrQPfeS8RMf9McuWl5WLhVXYhvS0vy -->
    </details>
    
    ---

### **5. Documentation & Compliance Refinement**

> **Brief Requirement (Section 4.5):** To "**Create a readme file to contain: ... AI transparency scale declaration statement [and] AI prompts (if used).**"
> <br>
> **Brief Requirement (Section 10):** To ensure the statement and log "**clearly communicates how you have used Artificial Intelligence.**"

*   <details>
    <summary><strong>Prompt 5.1: Crafting this AI Prompt Log</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt (Iterative Conversation):** This entry documents the iterative process of creating the AI Prompt Log itself. The conversation began with prompts like `"Help me create my AI prompt log"` and was followed by my own critical feedback and direction, such as:
        *   `"Nope doesn't work, too subjective. Quote the assignment so it's irrefutable."`
        *   `"The justification cannot be 'the brief said to do X, so I used AI to do X.' It should be 'The brief said I can use AI to do X.'"`
        *   `"It needs to be very clear... and use the two quotes from the AITS table in the brief."`
    *   **Outcome & Justification:** This interaction is a direct, provable example of the AITS-2 framework in action. The AI's role was to provide initial structures and drafts for this documentation. My role was to provide the critical direction to ensure the final output was not just a list, but a logically sound, evidence-based argument for compliance. **I refined and reviewed** every AI suggestion, rejecting flawed logic and demanding a higher standard of proof, which ultimately produced the clear and irrefutable log you are now reading. The full history of this meta-conversation is included in the `/AI-Prompts` folder as definitive evidence of this critical review process.
    <!-- Personal reference: https://aistudio.google.com/prompts/12BVdQD4ovVUd29QRjXosqcczrFUIyOIY -->
    </details>

*   <details>
    <summary><strong>Prompt 5.2: Auditing Log Accuracy</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt:** `"From this Chatbot transcript, which of the following prompt logs, is not mentioned? [Pasted full JSON chat history of this session]"`
    *   **Outcome & Justification:** I used the AI to cross-reference my drafted `README.md` against the actual chat history file to ensure 100% factual accuracy. The AI identified that the examples in Categories 2 and 3 (regarding `Conda` and `ImageDataGenerator`) were illustrative templates generated by the AI in previous turns, rather than actual prompts from this specific conversation thread. **I refined and reviewed** the final log based on this audit, ensuring that the submitted log only contains entries that are backed by the evidence in the provided transcript files.
    <!-- Personal reference: https://aistudio.google.com/prompts/1nXJa-g6ag-qopEYadFCOrfBP2n7WAuFF -->
    </details>

*   <details>
    <summary><strong>Prompt 5.3: Pre-Submission Review and Refinement</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output` / `Prompting Thinking`
    *   **My Prompt (Iterative Conversation):** After completing the codebase, I initiated a final review with prompts like: `"Look at the codebase and compare it to the assignment brief, then let me know what I'm missing."` This was followed by a series of specific, human-led refinement requests, including: `"But like... I can't actually see any results, perhaps we should 'display' rather than export it."`, `"Why not use a re-usable function...?"`
    *   **Outcome & Justification:** This interaction perfectly demonstrates the AITS-2 framework. The AI was used as a quality assurance partner to **improve the final output**. Its role was to generate templates (the `.ipynb` notebook), provide suggestions, and audit the code for minor issues. **I reviewed and refined** every single AI output. For instance:
        1. I corrected the AI's initial assumption about missing files.
        2. I identified a functional flaw in the AI-generated notebook (no visible results) and directed the fix.
        3. I proposed a superior coding practice (a reusable function) which the AI then helped implement, proving that the **human contribution was not just to review, but to actively improve upon the AI's suggestions**.
    *   The AI's contribution was clearly limited to "shaping" and "improving quality" under my direct supervision, with all critical thinking, problem identification, and final decisions being made by me.
    <!-- Personal reference: https://aistudio.google.com/prompts/1zsZoyABJK0AQ1e66dYfzbFP7micLahmw -->
    </details>