import os
import optax
import treescope

# Gemma imports
from kauldron import kd
from gemma import gm

if __name__ == "__main__":
    train_dialogue = list()
    for data_file in Path('./npc_dataset/train').glob('**/*'):
        with open(data_file, 'r', encoding='utf-8') as f:
            train_dialogue.extend(json.load(f))

    valid_dialogue = list()
    for data_file in Path('./npc_dataset/valid').glob('**/*'):
        with open(data_file, 'r', encoding='utf-8') as f:
            valid_dialogue.extend(json.load(f))

    training_data = [{'input_text': x['instruction'] + '\n' + x['context'], 'output_text': x['response']} for x in train_dialogue]
    valid_data = [{'input_text': x['instruction'] + '\n' + x['context'], 'output_text': x['response']} for x in valid_dialogue]

