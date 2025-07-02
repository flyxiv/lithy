import json
import os

from pathlib import Path
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import json

def get_normal_model(model_name):
    return AutoModelForCausalLM.from_pretrained(
         model_name,
    )


def get_quantized_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=[
        "query_key_value", 
        ],
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM", 
    )


    model = AutoModelForCausalLM.from_pretrained(
         model_name,
         torch_dtype=torch.float16,
         quantization_config=bnb_config
    )

    model = prepare_model_for_kbit_training(model)
    return model, lora_config


if __name__ == "__main__":
    train_dialogue = list()
    for data_file in Path('./npc_dataset/train').glob('**/*'):
        with open(data_file, 'r', encoding='utf-8') as f:
            train_dialogue.extend(json.load(f))

    valid_dialogue = list()
    for data_file in Path('./npc_dataset/valid').glob('**/*'):
        with open(data_file, 'r', encoding='utf-8') as f:
            valid_dialogue.extend(json.load(f))

    model_name = "EleutherAI/polyglot-ko-1.3b"
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=[
        "query_key_value", 
        ],
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM", 
    )


    model = get_normal_model(model_name) 

    max_length=2048
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length)
    print(tokenizer.eos_token_id)
    print(tokenizer.pad_token_id)

    train_dialogue_dict = {
        'instruction': [x['instruction'] for x in train_dialogue],
        'context': [x['context'] for x in train_dialogue],
        'response': [x['response'] for x in train_dialogue],
    }

    valid_dialogue_dict = {
        'instruction': [x['instruction'] for x in valid_dialogue],
        'context': [x['context'] for x in valid_dialogue],
        'response': [x['response'] for x in valid_dialogue],
    }


    train_data = Dataset.from_dict(train_dialogue_dict)
    valid_data = Dataset.from_dict(valid_dialogue_dict)

    train_data = train_data.map(
        lambda x: {'text': f"### 명령어: {x['instruction']}\n\n###맥락: {x['context']}\n\n### 답변: {x['response']}<|endoftext|>" }
    )
    valid_data = valid_data.map(
        lambda x: {'text': f"### 명령어: {x['instruction']}\n\n###맥락: {x['context']}\n\n### 답변: {x['response']}<|endoftext|>" }
    )

    train_data = train_data.map(lambda samples: tokenizer(samples["text"], padding='max_length'), batched=True)
    valid_data = valid_data.map(lambda samples: tokenizer(samples["text"], padding='max_length'), batched=True)


    columns_to_remove = ['instruction', 'context', 'response', 'text']
    train_data = train_data.remove_columns(columns_to_remove)
    valid_data = valid_data.remove_columns(columns_to_remove)

    tokenizer.padding_side = "right"
    model = get_peft_model(model, lora_config)
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="### 답변:",  
    )

    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        output_dir="./distributed_output",
        num_train_epochs=3,
        learning_rate=2e-5,
        save_steps=500,
        logging_steps=10,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_length=max_length,
        load_best_model_at_end=True,
        save_strategy='steps',
        eval_strategy='steps',
        eval_steps=10,
        label_names=['labels'],
        metric_for_best_model='eval_loss'
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model("./distributed_output/final_model")
