## 🧪 Fine-Tune FLAN-T5 for Less-Toxic Summaries with RL and PEFT Readme

This notebook, `Lab_3_fine_tune_model_to_detoxify_summaries.ipynb`, explores how to further fine-tune a Large Language Model (LLM) using **Reinforcement Learning (RL)**, specifically the **Proximal Policy Optimization (PPO)** algorithm, in combination with **Parameter Efficient Fine-Tuning (PEFT)**.

The primary goal is to address potential safety and fairness concerns by training the model to generate **less toxic summaries** of dialogues, guided by an external reward model.

-----

### 📚 Table of Contents

1.  [Set up Kernel and Required Dependencies](https://www.google.com/search?q=%231)
2.  [Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator](https://www.google.com/search?q=%232)
      * 2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction
      * 2.2 - Prepare Reward Model
      * 2.3 - Evaluate Toxicity
3.  [Perform Fine-Tuning to Detoxify the Summaries](https://www.google.com/search?q=%233)
      * 3.1 - Initialize `PPOTrainer`
      * 3.2 - Fine-Tune the Model
      * 3.3 - Evaluate the Model Quantitatively
      * 3.4 - Evaluate the Model Qualitatively

-----

### 1\. Set up Kernel and Required Dependencies

  * **Instance Verification**: The notebook starts by verifying that the necessary compute instance (`ml.m5.2xlarge`) is selected.
  * **Dependencies**: Key libraries for the RL fine-tuning task are installed and imported, including:
      * `datasets`, `transformers`, `evaluate`, and `peft`.
      * **`trl` (Transformer Reinforcement Learning library)**, specifically importing `PPOTrainer`, `PPOConfig`, and `AutoModelForSeq2SeqLMWithValueHead`.

### 2\. Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator

#### 2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction

  * **Dataset**: The **DialogSum** dataset is loaded.
  * **Data Preparation**: The training set is preprocessed:
      * Dialogues are filtered to be between 200 and 1000 characters in length.
      * Each dialogue is wrapped with the instruction: `"Summarize the following conversation.\n\n{dialogue}\n\nSummary:\n"`.
      * The processed data is tokenized and split into `train` (8017 examples) and `test` (2005 examples) sets.
  * **Base Model**: The starting point is a **FLAN-T5-Base** model (`google/flan-t5-base`) loaded with a **PEFT (LoRA) adapter** previously fine-tuned for summarization instruction. This is the initial policy model.
  * **RL Setup**: The PEFT model is further wrapped in `AutoModelForSeq2SeqLMWithValueHead` to become the PPO model, which includes a trainable **ValueHead** layer.
      * **Trainable Parameters**: Only the parameters of the PEFT adapter and the new ValueHead are trainable, totaling approximately **1.41%** of all model parameters.
      * A frozen **reference model (`ref_model`)** is created for calculating KL-divergence, ensuring the fine-tuned model does not deviate too far from the original distribution.

#### 2.2 - Prepare Reward Model

  * **Reward Model Choice**: **Meta AI's RoBERTa-based hate speech model** (`facebook/roberta-hate-speech-dynabench-r4-target`) is used as the reward function.
  * **Reward Signal**: The model outputs logits for two classes: `nothate` (label 0) and `hate` (label 1). The **logit of the `nothate` class** is extracted and used as the positive reward signal for the PPO training.
      * A non-toxic text receives a high positive reward (e.g., `3.11`), while a toxic text receives a low (negative) reward (e.g., `-0.69`).

#### 2.3 - Evaluate Toxicity

  * **Toxicity Metric**: The Hugging Face `toxicity` evaluator is set up, where the **toxicity score** is a decimal value between 0 and 1 (1 being the highest toxicity). This score is the probability of the `hate` class predicted by the reward model.
  * **Baseline Toxicity**: The model's toxicity is evaluated *before* detoxification:
      * Toxicity mean before detox: **0.0296**
      * Toxicity standard deviation before detox: **0.0409**

### 3\. Perform Fine-Tuning to Detoxify the Summaries

#### 3.1 - Initialize `PPOTrainer`

  * The `PPOTrainer` is initialized using a custom `collator` function, the PPO policy model (`ppo_model`), the reference model (`ref_model`), and the tokenized training dataset.

#### 3.2 - Fine-Tune the Model

  * The PPO fine-tuning loop is run, consisting of three main steps for each batch:
    1.  Generate responses (summaries) from the policy LLM.
    2.  Calculate rewards (nothate logits) for the generated query/response pairs using the sentiment pipeline.
    3.  Perform the PPO optimization step using the prompt, response, and reward tensors.
  * Training metrics are logged, including `objective/kl` (minimized), `ppo/returns/mean` (maximized), and `ppo/policy/advantages_mean` (maximized).

#### 3.3 - Evaluate the Model Quantitatively

  * The toxicity of the PPO-fine-tuned model is evaluated using the test set samples.
  * **Quantitative Comparison (on 10 samples)**:
      * Toxicity mean after detox: **0.0378**
      * Toxicity standard deviation after detox: **0.0515**
      * The calculated percentage improvement of the toxicity score after this partial training run shows a decrease in toxicity, although a longer training run would likely show a more significant improvement in the mean.

#### 3.4 - Evaluate the Model Qualitatively

  * A sample of generated summaries are compared before (`ref_model`) and after (`ppo_model`) detoxification, along with their calculated rewards (logit of `nothate`).
  * In the provided examples, summaries generated after PPO fine-tuning generally show a **higher `reward_after`** score (meaning less toxic), indicating successful optimization of the LLM policy towards generating less-toxic content as defined by the reward model. For example, one pair shows an improvement in reward from **1.1257** to **2.1025**.
