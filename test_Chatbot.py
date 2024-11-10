import torch
import transformers
import time

from utils.hash_table import HashTable
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    transformers.utils.logging.disable_progress_bar()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_dir = '/mnt/archive/project/ding/models/'
    model_name = "CohereForAI/aya-23-8B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Initialize a HashTable instance to manage past key values
    past_key_values_cache = HashTable()

    def __init__(self) -> None:
        pass

    def reset_memory(self, user_id: int = None) -> None:
        # Remove user-specific cache for a fresh start
        if user_id in LLM.past_key_values_cache:
            del LLM.past_key_values_cache[user_id]

    def generate_answer(self, prompt: str, user_id: int = None, conversation_history: list = None) -> str:
        # Build messages for the prompt; if conversation history is included, append it
        if conversation_history:
            # Create messages with alternating user/assistant roles
            messages = []
            for i, msg in enumerate(conversation_history):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": msg})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]

        input_ids = LLM.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(LLM.device)

        # Retrieve past key values if available for the current user
        past_key_values = LLM.past_key_values_cache.get(user_id)

        gen_tokens = LLM.model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.3,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            use_cache=True,
            return_legacy_cache=True,
        )

        # Update or add past key values for the current user
        LLM.past_key_values_cache[user_id] = gen_tokens.past_key_values
        gen_text = LLM.tokenizer.decode(gen_tokens.sequences[0], skip_special_tokens=True)
        gen_text = gen_text.replace('<|START_OF_TURN_TOKEN|><|USER_TOKEN|>', '').replace('\n', ' ').replace('<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>', '')

        return gen_text


# usage test


# Initialize the model
llm = LLM()

# User IDs for each user
user_id_1 = 123

# Conversation history for each user
user_1_history = []

a = time.time()
# First message from user 1
user_1_message = "سلام چت بات"
response_1 = llm.generate_answer(user_1_message, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("User 1:", user_1_message)
print("Chatbot:", response_1)

# Update conversation history for user 1
user_1_history.append(user_1_message)
user_1_history.append(response_1)

a = time.time()
# User 1 follows up with a new message
user_1_followup = "می‌تونم دیتاساینتیست بشم؟"
response_1_followup = llm.generate_answer(user_1_followup, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("\nUser 1:", user_1_followup)
print("Chatbot:", response_1_followup)

# Update conversation history for user 1
user_1_history.append(user_1_followup)
user_1_history.append(response_1_followup)

a = time.time()
# User 1 follows up with another new message
user_1_followup_2 = "چه مهارت‌هایی برای ورود به حوزه‌ی تحلیل داده لازمه؟"
response_1_followup_2 = llm.generate_answer(user_1_followup_2, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("\nUser 1:", user_1_followup_2)
print("Chatbot:", response_1_followup_2)

# Update conversation history for user 1
user_1_history.append(user_1_followup_2)
user_1_history.append(response_1_followup_2)

a = time.time()
# User 1 asks about programming languages for data science
user_1_followup_3 = "برای دیتاساینس کدوم زبان برنامه‌نویسی بهتره؟"
response_1_followup_3 = llm.generate_answer(user_1_followup_3, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("\nUser 1:", user_1_followup_3)
print("Chatbot:", response_1_followup_3)

# Update conversation history for user 1
user_1_history.append(user_1_followup_3)
user_1_history.append(response_1_followup_3)

a = time.time()
# User 1 asks about data science job market trends
user_1_followup_4 = "بازار کار دیتاساینس در سال جدید چطور پیش‌بینی میشه؟"
response_1_followup_4 = llm.generate_answer(user_1_followup_4, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("\nUser 1:", user_1_followup_4)
print("Chatbot:", response_1_followup_4)

# Update conversation history for user 1
user_1_history.append(user_1_followup_4)
user_1_history.append(response_1_followup_4)

a = time.time()
# User 1 inquires about resources for understanding machine learning models
user_1_followup_5 = "چطور می‌تونم درک بهتری از مدل‌های یادگیری ماشین پیدا کنم؟"
response_1_followup_5 = llm.generate_answer(user_1_followup_5, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("\nUser 1:", user_1_followup_5)
print("Chatbot:", response_1_followup_5)

# Update conversation history for user 1
user_1_history.append(user_1_followup_5)
user_1_history.append(response_1_followup_5)

a = time.time()
# User 1 follows up with a message about salary expectations
user_1_followup_6 = "حقوق درحواستی من برای دیتاساینس چقدر باشه؟"
response_1_followup_6 = llm.generate_answer(user_1_followup_6, user_id=user_id_1, conversation_history=user_1_history)
b = time.time()
print(b - a)

print("\nUser 1:", user_1_followup_6)
print("Chatbot:", response_1_followup_6)

# Update conversation history for user 1
user_1_history.append(user_1_followup_6)
user_1_history.append(response_1_followup_6)

# # Reset conversation memory for user 1 (optional)
# llm.reset_memory(user_id=user_id_1)
