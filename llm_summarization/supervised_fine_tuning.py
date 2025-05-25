import json
import os

from pathlib import Path
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


def get_summarize_prompt(file_path):
    with open(file_path, 'r', encoding='UTF8') as f:
        content = f.read()
        
    prompt = f"""
당신은 제공된 캐릭터 설명 텍스트를 분석하여 지정된 JSON 형식으로 핵심 정보를 요약하고 구조화하는 AI 어시스턴트입니다.

[캐릭터 설명 텍스트 시작]\n
"""
    prompt += content + '\n'
    prompt += """
[캐릭터 설명 텍스트 끝]

위 텍스트를 바탕으로 다음 JSON 형식에 맞춰 각 항목을 채워주세요. 각 필드에는 텍스트에서 관련된 핵심 정보만을 간결하게 요약하여 넣어주세요. 만약 특정 필드에 해당하는 내용이 텍스트에 명확히 언급되지 않았다면, 해당 필드 값은 null로 처리하거나 빈 배열([]) 또는 빈 문자열("")로 남겨주세요.

[요청 JSON 형식 시작]
{
  "character_id": "작품명_캐릭터명 format의 식별자 (String)",
  "character_name": "캐릭터 이름 (String)",
  "series_title": "작품 제목 (String)",
  "aliases": ["별명1 (String)", "다른 이름2 (String)"],
  "role_in_story": "스토리 내 역할 (String)",
  "archetype": ["원형1 (String)", "원형2 (String)"],
  "occupation_status": ["직업/신분1 (String)", "직업/신분2 (String)"],

  "appearance_summary": "외모 및 주요 신체적 특징에 대한 간략한 서술 (String)",
  "appearance_keywords": ["외모 키워드1 (String)", "외모 키워드2 (String)"],

  "personality_traits": ["성격 특성1 (String)", "성격 특성2 (String)"],
  "strengths": ["강점1 (String)", "강점2 (String)"],
  "weaknesses_flaws": ["약점/결점1 (String)", "약점/결점2 (String)"],
  "values_beliefs": ["가치관/신념1 (String)", "가치관/신념2 (String)"],
  "quirks_habits": ["특이점/습관1 (String)", "특이점/습관2 (String)"],

  "backstory_summary": "캐릭터의 배경 이야기 및 그를 형성한 주요 과거 경험에 대한 요약 (String)",
  "significant_life_events": [ // 캐릭터의 삶/성격에 전환점이 된 사건 (과거, 현재 포함)
    {
      "event_name": "결정적 사건명1 (String)",
      "event_description": "사건이 캐릭터에게 미친 영향 및 간략한 설명 (String)",
      "timeline_tag": "시기 (String)", // 예: "어린 시절", "ISSP 재직 중", "비밥 호 합류 직후"
      "related_characters": ["관련 캐릭터명1", "관련 캐릭터명2"]
    }
  ],

  "major_in_story_actions": [ // 작품의 주요 플롯 내에서 캐릭터가 수행한 행동/사건 (시간 순서나 중요도에 따라)
    {
      "arc_or_episode": "관련 에피소드/챕터/스토리 아크명 (String)", // 예: "세션 #5", "가니메데 비가 편", "붉은 눈의 악마 아크"
      "action_summary": "캐릭터의 주요 행동이나 역할에 대한 요약 (String)", // 예: "현상범 A를 추적하고 체포하는 데 결정적인 단서를 제공했다.", "동료 B를 구하기 위해 위험을 무릅썼다."
      "outcome_or_impact": "해당 행동의 결과나 스토리에 미친 영향 (String, Optional)", // 예: "현상금 획득에 기여", "캐릭터 C와의 관계가 깊어짐"
      "related_characters": ["관련 캐릭터명1", "관련 캐릭터명2"]
    }
  ],

  "motivations_goals": {
    "primary_motive": "가장 핵심적인 동기 (String)",
    "other_motives": ["기타 동기1 (String)", "기타 동기2 (String)"],
    "short_term_goals": ["단기 목표1 (String)", "단기 목표2 (String)"],
    "long_term_goals": ["장기 목표1 (String)", "장기 목표2 (String)"]
  },

  "relationships": [
    {
      "related_character_name": "관계 대상 캐릭터 이름 (String)",
      "relationship_type": "관계 유형 (String)",
      "relationship_description": "관계에 대한 간략한 설명 및 캐릭터에게 미치는 영향 (String)"
    }
  ],

  "abilities_skills": ["능력/기술1 (String)", "능력/기술2 (String)"],
  "key_quotes": ["대표적인 대사1 (String)", "대표적인 대사2 (String)"],
  "thematic_representation": ["상징하는 주제1 (String)", "상징하는 주제2 (String)"],
  "character_arc_summary": "캐릭터의 성장 및 변화 과정에 대한 요약 (String)",
  "tags_keywords": ["태그1 (String)", "태그2 (String)"]
}
[요청 JSON 형식 끝]

반드시 위 JSON 형식과 필드명을 정확히 따라서 결과를 생성해주세요.
    """

    return prompt

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"Prompt: {example['prompt'][i]}\nCompletion: {example['completion'][i]}" # 매우 기본적인 예시
        output_texts.append(text)
    return output_texts

if __name__ == "__main__":
    data_dir = './lithy_dataset/raw_text'
    label_dir = "./lithy_dataset/character_settings"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    prompts_list = list()
    completions_list = list()

    for txt_file in os.listdir(data_dir):
        if not txt_file.endswith('_character.txt'):
            continue

        file_path = Path(data_dir) / txt_file
        file_stem = file_path.stem
        json_file_path = Path(label_dir) / f"{file_stem}.json"

        if not file_path.exists() or not json_file_path.exists():
            continue

        prompts_list.append(get_summarize_prompt(file_path))
        with open(json_file_path, 'r', encoding='utf-8') as f:
            completion_content = json.load(f)
            completions_list.append(json.dumps(completion_content, ensure_ascii=False))

    train_dataset = Dataset.from_dict({
        'prompt': prompts_list,
        'completion': completions_list
    })

    model_name = "beomi/KoAlpaca-Polyglot-5.8B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        r=16, # LoRA 랭크 (작을수록 메모리 적게, 클수록 학습 능력 좋음)
        lora_alpha=32, # LoRA 스케일링 팩터
        target_modules=[
            "query_key_value", # 일반적으로 LLM의 QKV 레이어에 적용
        ],
        lora_dropout=0.05,
        bias="none", # 일반적으로 bias는 학습하지 않습니다.
        task_type="CAUSAL_LM", # 텍스트 생성 모델이므로 CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        output_dir="./distributed_output",
        num_train_epochs=500,
        learning_rate=2e-4,
        fp16=True,
        save_steps=500,
        logging_steps=10,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()
    trainer.save_model("./distributed_output/final_model")