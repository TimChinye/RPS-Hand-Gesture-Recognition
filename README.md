# Rock-Paper-Scissors Hand Gesture Recognition System

Rest of README...

## AI Transparency Statement

**AITS Descriptor:** AITS 2: AI for Shaping

For this assessment, I have used Artificial Intelligence (AI) in accordance with AITS 2 (AI for Shaping) of the Artificial Intelligence Transparency Scale. This statement is an expansion of [the university's exemplar](https://www.shu.ac.uk/myhallam/study/assessment/artificial-intelligence-and-assessment#:~:text=an%20example%20student%20statement), adapted for a technical, code-based assessment rather than a traditional essay, whilst staying in accordance with the guidance provided in the assessment brief.

Specifically, I used Google's Gemini (via [AI Studio](https://aistudio.google.com/)) for the following "shaping" activities:

--- Start: To be updated, for accuracy.

*   **Conceptual Clarification:** To ask for explanations of complex TensorFlow and Keras concepts, such as the specific parameters in convolutional layers or the mathematical difference between loss functions. This was used in a similar way to consulting a textbook or official documentation.
*   **Debugging Assistance:** To help identify the cause of isolated, specific errors in my code. For example, I provided small code snippets that were producing `TensorShape` mismatch errors to get suggestions on potential causes, which I then investigated and fixed myself.
*   **Structural Refinement:** To get suggestions on improving the structure and clarity of documentation. This included asking for feedback on the layout of the `README.md` file and rephrasing in-code comments to be more professional and understandable.

--- End: To be updated, for accuracy.

My own human contribution was central and a major part to the entirety of the project's core decisions and technical work. **AI-generated suggestions were treated as a starting point for my own investigation and were never directly implemented without critical evaluation and adaptation.** My contributions include:

*   The complete design of both neural network architectures (the model built from scratch and the transfer learning model).
*   The implementation of the entire data collection, preprocessing, and augmentation pipeline.
*   The writing of the complete training, validation, and performance evaluation scripts.
*   The design and implementation of the "Additional AI Capability" (the smart AI player).
*   The critical analysis and interpretation of all model performance results, forming the conclusions about their effectiveness.

In summary, I used AI to learn through a series of questions and answers and as a debugging tool to shape my approach and overcome minor technical hurdles. It did not contribute to the algorithmic design, implementation, or analytical conclusions of this project. All substantive code, architectural decisions, and analytical insights are entirely my own.

## AI Prompt Log Summary

This section provides a curated log of the key prompts used to shape this project. The unabridged chat histories, including the full AI responses, are provided in the `/AI_Prompts` folder included at the root of the submitted folder.

<details>
<summary><strong>1. Structural & Documentation Refinement</strong></summary>

These prompts were used to ensure the project's structure and documentation met professional standards.

*   **Prompt:**
    > "Okay, working on this. What should the project structure look like... [followed by details]"
*   **How the Response Was Used:**
    The AI provided a detailed template for a professional project structure, separating `notebooks` for experimentation from `src` for application code. I adopted this structure, including creating sub-packages like `src/models` and `src/utils` to organize my code logically and demonstrate best practices.

</details>

<details>
<summary><strong>2. Conceptual Clarification & Learning</strong></summary>

These prompts were used to make informed decisions about tools and techniques.

*   **Prompt:**
    > "Venv vs Conda? What's the difference and which is more appropiate for me"
*   **How the Response Was Used:**
    The AI provided a detailed comparison table and an analogy. Based on its explanation that Conda excels at managing complex non-Python dependencies (critical for TensorFlow and OpenCV on Windows), I made the core decision to switch from `venv` to `Conda`, which ultimately resolved my setup issues.

*   **Prompt:**
    > "\"Use Keras's ImageDataGenerator...\" Is this included in the 250 * 4 images? Or is this used to create a total greater than 1000 images, as in 4000+."
*   **How the Response Was Used:**
    The AI's response, including the "teaching a child" analogy, clarified the crucial distinction between the physically captured dataset (~1000 images) and the virtually augmented data generated on-the-fly during training. This understanding was fundamental to implementing my training pipeline correctly.

</details>

<details>
<summary><strong>3. Debugging Assistance</strong></summary>

These prompts were used to troubleshoot persistent environment setup failures on Windows.

*   **Prompt:**
    > "(venv) C:\\...\\AI_&_Machine_Learning_2>deactivate\n'deactivate' is not recognized as an internal or external command"
*   **How the Response Was Used:**
    The AI diagnosed this as a "zombie" or partially activated environment where the system `PATH` was not correctly updated. Its advice to close the terminal completely and start a fresh session was a key step in breaking the troubleshooting loop and moving toward a clean setup.

*   **Prompt:**
    > "CondaToSNonInteractiveError: Terms of Service have not been accepted..."
*   **How the Response Was Used:**
    The AI immediately identified this not as a technical bug, but as a one-time licensing step required by Anaconda. It provided the exact three `conda tos accept` commands I needed to run in the Anaconda Prompt, which unblocked the environment creation process in VS Code.

</details>

<details>
<summary><strong>4. Compliance & Best Practice Verification</strong></summary>

This prompt was used to ensure that my method of documenting AI usage was itself compliant with the university's academic integrity policies.

*   **Prompt:**
    > "What AI prompts should I provide exactly? Should I give them these prompts? As in the ones I'm talking to you with right now..."
*   **How the Response Was Used:**
    The AI's guidance on this meta-question was crucial. It confirmed that including strategic planning and debugging prompts was not only compliant but also a powerful way to demonstrate a mature, professional use of AI under AITS 2. This shaped the final structure of this prompt log.

*   **Prompt:**
    > "Are the following conversations in compliant with the AITS Level 2?:"
*   **How the Response Was Used:**
    The AI's analysis of our conversation confirmed that my use of it for project planning, conceptual clarification, and debugging fell squarely within the definition of AITS Level 2. This provided the confidence to be fully transparent about my development process, framing the AI interaction as a professional and ethical use of a modern tool rather than something to be concealed.

</details>