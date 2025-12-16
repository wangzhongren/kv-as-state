# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_DIR = "./Qwen/Qwen3-0.6B"
DTYPE = torch.float16

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7

print("🔁 加载模型...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

model.eval()
print("✅ 模型加载完成！")

# ======================
# 保存和加载 KV 状态的函数
# ======================

def save_past_kv(past_kv, filename='past_kv.pth'):
    if past_kv is not None:
        torch.save(past_kv, filename)
        print(f"✅ KV 状态已保存到 {filename}")

def load_past_kv(filename='past_kv.pth'):
    if os.path.exists(filename):
        past_kv = torch.load(filename)
        # 将所有张量移动到模型设备（递归处理元组/列表结构）
        def move_to_device(item):
            if isinstance(item, torch.Tensor):
                return item.to(model.device)
            elif isinstance(item, tuple):
                return tuple(move_to_device(i) for i in item)
            elif isinstance(item, list):
                return [move_to_device(i) for i in item]
            return item
        past_kv = move_to_device(past_kv)
        print(f"✅ KV 状态已从 {filename} 加载")
        return past_kv
    else:
        print(f"⚠️ 文件 {filename} 不存在，使用 None 作为初始 KV")
        return None

# ======================
# 连续 KV 状态生成函数
# ======================

@torch.no_grad()
def generate_continuous(prompt, past_kv=None):
    input_text = build_chat_prompt(prompt)

    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(model.device)

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_kv,
        use_cache=True
    )

    past_kv = outputs.past_key_values
    logits = outputs.logits[:, -1, :] / TEMPERATURE

    generated = input_ids.clone()

    for _ in range(MAX_NEW_TOKENS):
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_id], dim=1)

        outputs = model(
            input_ids=next_id,
            past_key_values=past_kv,
            use_cache=True
        )

        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :] / TEMPERATURE

        if next_id.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(
        generated[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return text.strip(), past_kv

# ======================
# 构建聊天提示函数
# ======================

def build_chat_prompt(user_text):
    messages = [
        {"role": "system", "content": "你是一个简洁、准确的中文助手。"},
        {"role": "user", "content": user_text}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# ======================
# 测试（支持保存和重放）
# ======================

# 先加载已保存的 KV（如果存在）
past_kv = load_past_kv('past_kv.pth')  # 可以自定义文件名

dialogs = [
    "巴黎是哪个国家的首都？",
    "它有哪些著名景点？",
    "请用一句话赞美它。"
]

# dialogs = [
#     "它的人口大约有多少？",
#     "为什么说它是浪漫之都？",
#     "给我推荐一个巴黎的三天旅行行程。"
# ]

for i, q in enumerate(dialogs, 1):
    print(f"\n=== 对话 {i} ===")
    print(q)
    ans, past_kv = generate_continuous(q, past_kv)
    print(ans)
    # 可选：在每个对话后保存 KV
    # save_past_kv(past_kv, f'past_kv_turn{i}.pth')

# 所有对话结束后保存 KV（用于下次重放）
save_past_kv(past_kv, 'past_kv.pth')