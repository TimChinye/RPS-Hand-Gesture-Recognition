# Rock-Paper-Scissors Hand Gesture Recognition System

Rest of README...

## AI Transparency Statement

**AITS Descriptor:** AITS 2: AI for Shaping

In accordance with the assessment brief (Section 10), I have used Artificial Intelligence at the permitted level of AITS 2. The university's AITS table defines this level with two key statements:

1.  **Permitted AI Contribution:** AI can be used for "shaping parts of the activity. This includes **initial outlining, concept development, prompting thinking, and/or improving structure/quality of the final output.**"
2.  **Required Human Contribution:** "Most of the activity is human developed/generated. **AI ideas and suggestions are refined and reviewed.**"

I pledge that my use of AI adheres strictly to this framework. The following log provides irrefutable evidence of this process, demonstrating how each interaction was a permitted "shaping" activity and how every AI suggestion was critically "refined and reviewed" by me to meet the specific demands of the assessment brief.

The unabridged chat histories are provided in the `/AI_Prompts` folder.

## AI Prompt Log: Evidence of AITS-2 Compliance

---

### **1. Initial Outlining & Project Structuring**

> **Brief Requirement (Section 1):** To create a "**fully functional neural network system**... ensuring that your approach is both theoretically sound and practically applicable."
>
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
    <!-- Personal reference: [INSERT A NEW PERSONAL REFERENCE LINK TO THIS SPECIFIC CHAT] -->
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
    *   **Outcome & Justification:** This interaction was a classic example of using AI for `prompting thinking`. The AI's initial advice was based on common best practices (e.g., using underscores). My critical feedback (`"Underscore failed..."`) forced a deeper analysis. The AI's subsequent explanation of *why* the `&` character was causing the `LinkError` was a key moment of `concept development`. **I refined and reviewed** this information, which confirmed my own diagnosis. This allowed me to confidently implement the solution (renaming the folder), which was essential for creating the required "**error free and ready to run**" code.
    <!-- Personal reference: https://aistudio.google.com/prompts/1o38kUHxwNLqDyrGmN1Yat39jhEDyczPV -->
    </details>


---

### **4. Code Generation for Boilerplate Tasks**

> **Brief Requirement (Section 3.1):** To "**write suitable python code to capture several images per hand gestures category... through computer’s camera.**"
>
> **Brief Requirement (LO3):** To "**implement... artificial neural networks to solve real-world artificial intelligent problems.**"

*   <details>
    <summary><strong>Prompt 4.1: Generating the Data Collection Script</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt:** `"Write a Python script using OpenCV to capture images from your webcam. The script should save images to structured folders (/dataset/rock, /dataset/paper, etc.). Here's my existing project structure: [pasted project tree structure]"`
    *   **Outcome & Justification:** In line with AITS-2, I used AI to generate a boilerplate script for a standard, repetitive task. This improved the quality and speed of my initial setup. After generating three versions, **I refined and reviewed** them extensively, combined specific ideas and refactored it to make it m own.
    *   The AI's role was limited to shaping the initial tool; the core intellectual work of creating the dataset, a key part of solving this "real-world problem," was entirely my own.
    <!-- Personal reference: https://aistudio.google.com/prompts/1o38kUHxwNLqDyrGmN1Yat39jhEDyczPV -->
    </details>
    
    ---

### **5. Documentation & Compliance Refinement**

> **Brief Requirement (Section 4.5):** To "**Create a readme file to contain: ... AI transparency scale declaration statement [and] AI prompts (if used).**"
>
> **Brief Requirement (Section 10):** To ensure the statement and log "**clearly communicates how you have used Artificial Intelligence.**"

*   <details>
    <summary><strong>Prompt 5.1: Crafting this AI Prompt Log</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt (Iterative Conversation):** This entry documents the iterative process of creating the AI Prompt Log itself. The conversation began with prompts like `"Help me create my AI prompt log"` and was followed by my own critical feedback and direction, such as:
        *   `"Nope doesn't work, too subjective. Quote the assignment so it's irrefutable."`
        *   `"The justification cannot be 'the brief said to do X, so I used AI to do X.' It should be 'The brief said I can use AI to do X.'"`
        *   `"It needs to be very clear... and use the two quotes from the AITS table in the brief."`
    *   **Outcome & Justification:** This interaction is a direct, provable example of the AITS-2 framework in action. The AI's role was to provide initial structures and drafts for this documentation. My role was to provide the critical direction to ensure the final output was not just a list, but a logically sound, evidence-based argument for compliance. **I refined and reviewed** every AI suggestion, rejecting flawed logic and demanding a higher standard of proof, which ultimately produced the clear and irrefutable log you are now reading. The full history of this meta-conversation is included in the `/AI_Prompts` folder as definitive evidence of this critical review process.
    <!-- Personal reference: https://aistudio.google.com/prompts/12BVdQD4ovVUd29QRjXosqcczrFUIyOIY -->
    </details>

*   <details>
    <summary><strong>Prompt 4.2: Auditing Log Accuracy</strong></summary>

    *   **AITS-2 Activity:** `Improving Structure/Quality of the Final Output`
    *   **My Prompt:** `"From this Chatbot transcript, which of the following prompt logs, is not mentioned? [Pasted full JSON chat history of this session]"`
    *   **Outcome & Justification:** I used the AI to cross-reference my drafted `README.md` against the actual chat history file to ensure 100% factual accuracy. The AI identified that the examples in Categories 2 and 3 (regarding `Conda` and `ImageDataGenerator`) were illustrative templates generated by the AI in previous turns, rather than actual prompts from this specific conversation thread. **I refined and reviewed** the final log based on this audit, ensuring that the submitted log only contains entries that are backed by the evidence in the provided transcript files.
    <!-- Personal reference: https://aistudio.google.com/prompts/1nXJa-g6ag-qopEYadFCOrfBP2n7WAuFF -->
    </details>