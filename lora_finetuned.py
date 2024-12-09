from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from datasets import load_dataset

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",  
    r=16,                   
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.1,       
    target_modules=["q_proj", "v_proj"]  
)

model = get_peft_model(model, lora_config)

dataset = load_dataset('csv', data_files='ifqa_main.csv')
dataset = dataset["train"].train_test_split(test_size=0.2)
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples["Input"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

model.save_pretrained("lora/lora_model")
tokenizer.save_pretrained("lora/lora_model")