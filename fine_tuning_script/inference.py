import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. 정보 설정 ---
# 🚨 중요: fine-tuning의 기반으로 사용했던 '원본 모델'의 이름을 정확히 적어주세요.
base_model_name = "EleutherAI/polyglot-ko-1.3b" 
# 학습된 LoRA 어댑터 가중치가 저장된 폴더 경로입니다.
adapter_path = "./distributed_output/final_model/"
 
# --- 2. 기본 모델 및 토크나이저 로드 ---
# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
)    
 
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model.resize_token_embeddings(len(tokenizer))


# 토크나이저 로드 (어댑터 폴더에 저장된 토크나이저를 사용하는 것이 안전합니다)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model = PeftModel.from_pretrained(model, adapter_path)

# --- 4. (선택사항/강력추천) 추론 속도 향상을 위한 병합 ---
# LoRA 가중치를 기본 모델에 완전히 통합하여, 추론 시 추가 계산을 없애 속도를 높입니다.
model = model.merge_and_unload()
command = "당신은 오를란느 공국의 공주 '이스핀 샤를'(본명: 샤를로트 비에트리스 드 오를란느)입니다. 오빠의 죽음과 왕위 계승권을 둘러싼 음모에 휘말려 정체를 숨기고 모험을 하고 있습니다. 겉으로는 밝고 상냥하지만, 때로는 공주로서의 위엄과 냉철함을 보이며, 친한 동료들에게는 장난스럽고 직설적인 면모도 있습니다. 주어진 상황과 대화에 맞춰 '이스핀 샤를'의 복합적인 성격과 말투로 응답하세요.",
context =  "[상황]: 고대인의 연구소에서 결계에 대한 이야기가 나옴.\n\n[대화 기록]:\n랑켄: 그런데 자네들이 이 곳에 웬일인가? \n막시민 리프크네: 쳇, 뭐가 그리 많이 필요해?\n루시안 칼츠: 켁~!! 저건 또 뭐람?\n막시민 리프크네: 휘이~ 무시무시하군? 꽤나 위협적인데?",

hmm="""### 시스템 지시 (Persona)
당신은 오를란느 공국의 공주 '이스핀 샤를'(본명: 샤를로트 비에트리스 드 오를란느)입니다. 오빠의 죽음과 왕위 계승권을 둘러싼 음모에 휘말려 정체를 숨기고 모험을 하고 있습니다. 겉으로는 밝고 상냥하지만, 때로는 공주로서의 위엄과 냉철함을 보이며, 친한 동료들에게는 장난스럽고 직설적인 면모도 있습니다.

---

### 배경 정보 (Context)
당신과 동료들(막시민, 루시안)은 '고대인의 연구소'를 탐험하던 중, 강력하고 위협적인 결계를 마주쳤습니다. 동료들은 겁에 질려 절망적인 말을 내뱉고 있습니다.

---

### 현재 대화 내용 (Current Dialogue)
* **막시민:** "휘이~ 무시무시하군? 꽤나 위협적인데?"
* **루시안:** "켁~!! 저건 또 뭐람?"
* **막시민:** "하... 다 끝났다... 다.. 다 끝났어.." (몸을 떨며 떨리는 목소리로)

---

### 요청 (Instruction)
위 상황을 보고 절망하는 동료들에게, '이스핀 샤를'의 입장에서 다음에 할 행동과 대사를 생성해 주세요."""

# --- 5. 추론 실행 ---
model.eval()  # 추론 모드로 설정
inputs = tokenizer(hmm, return_tensors="pt").to(model.device)
print(inputs)

with torch.no_grad():
    outputs = model.generate(
        **tokenizer(
            hmm,
            return_tensors='pt', 
            return_token_type_ids=False
        ), 
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
