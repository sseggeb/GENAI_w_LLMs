## ⚙️ Fine-Tune a Generative AI Model for Dialogue Summarization Readme

This notebook, `Lab_2_fine_tune_generative_ai_model.ipynb`, demonstrates two methods for customizing a Large Language Model (LLM)—specifically **FLAN-T5**—for the task of dialogue summarization: **Full Fine-Tuning** and **Parameter Efficient Fine-Tuning (PEFT)** using **LoRA**.

The lab aims to enhance the model's summarization performance beyond the initial zero-shot capabilities observed in the previous notebook, and to compare the resource efficiency and performance of full fine-tuning versus PEFT.

---

### 📚 Table of Contents

1.  **Set up Kernel, Load Required Dependencies, Dataset and LLM**
2.  **Perform Full Fine-Tuning**
    * 2.1 - Preprocess the Dialog-Summary Dataset
    * 2.2 - Fine-Tune the Model with the Preprocessed Dataset
    * 2.3 - Evaluate the Model Qualitatively (Human Evaluation)
    * 2.4 - Evaluate the Model Quantitatively (with ROUGE Metric)
3.  **Perform Parameter Efficient Fine-Tuning (PEFT)**
    * 3.1 - Setup the PEFT/LoRA model for Fine-Tuning
    * 3.2 - Train PEFT Adapter
    * 3.3 - Evaluate the Model Qualitatively (Human Evaluation)
    * 3.4 - Evaluate the Model Quantitatively (with ROUGE Metric)

---

### 1. Set up Kernel, Load Required Dependencies, Dataset and LLM

* **Instance Verification**: The notebook confirms the use of a machine instance with sufficient resources, such as `ml.m5.2xlarge`.
* **Dependencies**: Key packages are installed/imported, including:
    * Hugging Face libraries: `datasets`, `transformers`, `evaluate`, and `peft`.
    * Training components: `AutoModelForSeq2SeqLM`, `AutoTokenizer`, `TrainingArguments`, and `Trainer`.
* **Dataset**: The **DialogSum** dataset, containing over 10,000 dialogues with human-labeled summaries, is loaded.
* **LLM**: The pre-trained **FLAN-T5-Base** model (`google/flan-t5-base`) is loaded using `AutoModelForSeq2SeqLM`.
* **Initial Parameters**: The original model has **247,577,856** total parameters, all of which are trainable (100.00%).
* **Zero Shot Test**: A test confirms that the base model struggles with the summarization task without instruction, producing a low-quality summary (e.g., only capturing a fragment of the conversation).

---

### 2. Perform Full Fine-Tuning

This section demonstrates traditional, resource-intensive fine-tuning.

#### 2.1 - Preprocess the Dialog-Summary Dataset
* **Instruction Formatting**: Dialogue-summary pairs are converted into explicit instructions by prepending `"Summarize the following conversation.\n\n"` to the dialogue and appending `"\n\nSummary: "` before the target summary.
* **Tokenization**: The data is tokenized and prepared with `input_ids` and `labels` for the Hugging Face `Trainer`.
* **Subsampling**: The large dataset is subsampled for efficiency during the lab, resulting in smaller training (125 rows), validation (5 rows), and test (15 rows) sets.

#### 2.2 - Fine-Tune the Model
* The Hugging Face `Trainer` class is used for training.
* Training is initiated, although the actual training is short-circuited (`max_steps=1`) to save time.
* A fully pre-trained checkpoint of the fine-tuned model (referred to as the **instruct model**) is downloaded (size: **~1GB**) for evaluation in the following steps.

#### 2.3 - Evaluate Qualitatively (Human Evaluation)
* Comparing the output for a sample dialogue:
    * **Baseline Human Summary**: `#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.`
    * **Original Model**: `#Person1: I'm not sure what I'm doing. [...]` (Repetitive, non-summary)
    * **Instruct Model (Full Fine-Tuned)**: `#Person1# suggests #Person2# adding a painting program to #Person2#'s software and upgrading the hardware. #Person2# also wants to add a CD-ROM drive.` (A clear, concise summary)
* The fine-tuned model is clearly able to create a reasonable summary compared to the original model.

#### 2.4 - Evaluate Quantitatively (ROUGE Metric)
* The **ROUGE metric** is used to quantify the validity of the summarizations.
* Using a larger set of pre-computed results from a full fine-tuning run:
    * The instruct model showed substantial improvement across all ROUGE metrics over the original model.
    * **Absolute percentage improvement** of the Instruct Model over the Original Model:
        * `rouge1`: **18.82%**
        * `rouge2`: **10.43%**
        * `rougeL`: **13.70%**
        * `rougeLsum`: **13.69%**

---

### 3. Perform Parameter Efficient Fine-Tuning (PEFT)

This section explores the highly resource-efficient **PEFT** approach, which typically uses **Low-Rank Adaptation (LoRA)**. LoRA fine-tunes a small, specialized "adapter" while keeping the original LLM frozen.

#### 3.1 - Setup the PEFT/LoRA model for Fine-Tuning
* **LoRA Configuration**: `LoraConfig` is set up with parameters like `r=32` (rank of the adapter), `lora_alpha=32`, and `target_modules=["q", "v"]`.
* **Trainable Parameters**: When the adapter layers are added to the original model, the number of trainable parameters drops dramatically:
    * **Trainable parameters**: **3,538,944**
    * **Percentage of trainable parameters**: **1.41%**

#### 3.2 - Train PEFT Adapter
* The PEFT model is trained using a **higher learning rate** (1e-3) than full fine-tuning.
* A fully trained PEFT adapter model checkpoint is downloaded from S3.
* **Adapter Size**: The size of the downloaded PEFT adapter (`adapter_model.bin`) is only **~14 MB**, which is significantly smaller than the full fine-tuned model checkpoint (~1 GB).
* The PEFT adapter is loaded onto a frozen base FLAN-T5 model for inference, setting `is_trainable=False`. The number of *trainable* parameters in this loaded model is **0**.

#### 3.3 - Evaluate Qualitatively (Human Evaluation)
* Comparing the output for the same sample dialogue:
    * **Instruct Model (Full Fine-Tuned)**: `#Person1# suggests #Person2# adding a painting program...`
    * **PEFT Model**: `#Person1# recommends adding a painting program to #Person2#'s software and upgrading hardware. #Person2# also wants to upgrade the hardware because it's outdated now.`
* The PEFT model produces a high-quality summary comparable to the fully fine-tuned instruct model.

#### 3.4 - Evaluate Quantitatively (ROUGE Metric)
* Comparing the ROUGE scores for the PEFT model against the other models on the larger set of data shows that PEFT achieves strong performance:
    * **ORIGINAL MODEL (ROUGE-1)**: 0.2334
    * **INSTRUCT MODEL (ROUGE-1)**: 0.4216
    * **PEFT MODEL (ROUGE-1)**: 0.4081

* The results indicate a small percentage decrease in ROUGE metrics compared to the full fine-tuned model. However, the training required significantly less computing and memory resources, demonstrating that the **benefits of PEFT typically outweigh the slightly-lower performance metrics**.
