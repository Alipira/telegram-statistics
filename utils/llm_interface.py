import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hash_table import HashTable


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
