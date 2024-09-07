# Medical Chatbot App

This repository contains a **Medical Chatbot App** powered by a fine-tuned **DistilBERT-GPT2** model. The chatbot is designed to provide medical information and answer general health-related questions. It uses a hybrid model architecture, where **DistilBERT** acts as the encoder for understanding the user queries and **GPT-2** generates the relevant responses.

## Features

- **Medical Query Understanding**: The chatbot is designed to handle various medical queries and provide meaningful responses.
- **Pre-trained Model**: The chatbot is trained on a custom dataset using the DistilBERT and GPT-2 models for efficient and accurate predictions.
- **Easy Deployment**: Can be easily deployed as a web app using frameworks like **Streamlit**.
- **Interactive UI**: Simple and interactive user interface that allows users to enter queries and get instant responses.

## Tech Stack

- **Model Architecture**: DistilBERT + GPT-2
- **Fine-Tuning Framework**: Hugging Face Transformers
- **Deployment**: Streamlit for UI
- **Languages**: Python
- **Libraries**: 
  - Transformers (Hugging Face)
  - PyTorch
  - Datasets (for data handling)
  - Streamlit (for web app UI)

## Installation

To run the chatbot locally, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-chatbot-app.git
cd medical-chatbot-app
```

### 2. Install Dependencies

Ensure you have Python installed (preferably 3.8 or higher), then install the necessary libraries using:

```bash
pip install -r requirements.txt
```

### 3. Download or Load the Pre-trained Model

To use the chatbot, ensure you download the fine-tuned `distilbert-gpt2` model.

You can load the model directly from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-model-name")
model = AutoModelForCausalLM.from_pretrained("your-model-name")
```

### 4. Run the App

Once the dependencies are installed, you can launch the Streamlit app using:

```bash
streamlit run app.py
```

This will start the app locally on your machine, and you can access it through `http://localhost:8501`.

## Model Training

The chatbot is fine-tuned on a medical QA dataset using the following steps:

1. **Dataset**: We use the `viniljpk/pubmedQA_v1` dataset for fine-tuning the model.
2. **Tokenizer**: Tokenize the data using the `distilbert-base-uncased` tokenizer.
3. **Training**: Fine-tune the `distilbert-gpt2` model for medical query generation.
4. **Evaluation**: Evaluate the model using accuracy and perplexity metrics on the test set.

### Fine-Tuning Code

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSeq2SeqLM.from_pretrained("distilbert-gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=10,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()
```

## Usage

Once the app is running, users can enter medical questions like:

- "What are the symptoms of diabetes?"
- "How to treat high blood pressure?"
- "What are the side effects of ibuprofen?"

The chatbot will generate a response based on its training and knowledge base.

## Future Improvements

- **Contextual Understanding**: Improve the chatbot's ability to handle long conversations with better context retention.
- **Multilingual Support**: Add support for multiple languages to broaden accessibility.
- **Improved Accuracy**: Fine-tune the model further with more specialized medical datasets to improve accuracy in responses.

## Contributing

We welcome contributions! Please feel free to raise issues, fork the repo, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---