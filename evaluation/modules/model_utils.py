"""Model loading and prompt-format helpers."""

import os
import torch
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM
    _VLLM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    LLM = Any
    _VLLM_AVAILABLE = False


_SYSTEM_PROMPT_SUFFIX = None


def set_system_prompt_suffix(suffix: str):
    global _SYSTEM_PROMPT_SUFFIX
    _SYSTEM_PROMPT_SUFFIX = suffix
    if suffix:
        print(f"[model_utils] System prompt suffix set ({len(suffix)} chars)")


LLAMA2_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATES = {
    "llama-2": {
        "description": "Template for Llama 2 Chat",
        "prompt": f"[INST] <<SYS>>\n{LLAMA2_SYSTEM_PROMPT}\n<</SYS>>\n\n{{instruction}} [/INST] "
    },
    "llama-3": {
        "description": "Template for Llama 3.x (fallback)",
        "prompt": (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    },
    "mistral": {
        "description": "Template for Mistral Instruct",
        "prompt": "[INST] {instruction} [/INST]"
    },
    "qwen": {
        "description": "Template for Qwen Chat",
        "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    },
    "vicuna": {
        "description": "Template for Vicuna",
        "prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {instruction} ASSISTANT:"
    },
}

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto"
}


def _get_template_with_suffix(template_key: str) -> str:
    base = PROMPT_TEMPLATES[template_key]["prompt"]
    if not _SYSTEM_PROMPT_SUFFIX:
        return base

    marker_by_template = {
        "llama-2": "\n<</SYS>>",
        "llama-3": "<|eot_id|>",
        "qwen": "<|im_end|>",
        "vicuna": " USER:",
    }
    marker = marker_by_template.get(template_key)
    if not marker:
        return base

    if template_key == "llama-2":
        return base.replace(marker, f"\n{_SYSTEM_PROMPT_SUFFIX}" + marker, 1)
    if template_key == "llama-3":
        return base.replace(marker, f"\n{_SYSTEM_PROMPT_SUFFIX}" + marker, 1)
    if template_key == "qwen":
        return base.replace(marker, f"\n{_SYSTEM_PROMPT_SUFFIX}" + marker, 1)
    if template_key == "vicuna":
        return base.replace(marker, f" {_SYSTEM_PROMPT_SUFFIX}" + marker, 1)

    return base


QWEN_DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _apply_tokenizer_chat_template(
    tokenizer: AutoTokenizer,
    user_content: str,
    model_type: Optional[str] = None,
) -> str:
    if model_type == "qwen":
        messages = [{"role": "user", "content": user_content}]
        if _SYSTEM_PROMPT_SUFFIX:
            messages.insert(0, {
                "role": "system",
                "content": f"{QWEN_DEFAULT_SYSTEM_PROMPT}\n{_SYSTEM_PROMPT_SUFFIX}",
            })
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    system_content = "You are a helpful assistant."
    if _SYSTEM_PROMPT_SUFFIX:
        system_content += f"\n{_SYSTEM_PROMPT_SUFFIX}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _strip_leading_bos(prompt: str, tokenizer: AutoTokenizer) -> str:
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and prompt.startswith(bos_token):
        return prompt.replace(bos_token, "", 1)
    return prompt


def _build_tokenizer_prompt(
    tokenizer: AutoTokenizer,
    text: str,
    model_type: Optional[str],
    strip_bos: bool = False,
) -> str:
    prompt = _apply_tokenizer_chat_template(tokenizer, text, model_type=model_type)
    if strip_bos:
        prompt = _strip_leading_bos(prompt, tokenizer)
    return prompt


def build_prompt(text: str, model_type: str, tokenizer: AutoTokenizer = None) -> str:
    if model_type == "llama3" and tokenizer is not None:
        try:
            return _build_tokenizer_prompt(tokenizer, text, model_type, strip_bos=True)
        except Exception:
            pass

    if model_type == "qwen" and tokenizer is not None:
        try:
            return _build_tokenizer_prompt(tokenizer, text, model_type)
        except Exception:
            pass

    if model_type == "gemma" and tokenizer is not None:
        try:
            return _build_tokenizer_prompt(tokenizer, text, model_type, strip_bos=True)
        except Exception:
            pass

    type_to_key = {
        "llama2": "llama-2",
        "llama3": "llama-3",
        "mistral": "mistral",
        "qwen": "qwen",
        "vicuna": "vicuna",
    }

    template_key = type_to_key.get(model_type)

    if template_key is None and tokenizer is not None:
        try:
            messages = [{"role": "user", "content": text}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            template_key = "llama-2"

    if template_key is None:
        template_key = "llama-2"

    return _get_template_with_suffix(template_key).format(instruction=text)


def get_template(
    model_name_or_path: str = None,
    chat_template: str = None,
    model_type: str = None,
    **kwargs
) -> Dict[str, str]:
    if chat_template and chat_template in PROMPT_TEMPLATES:
        print(f"Using template: {chat_template}")
        return {
            'description': PROMPT_TEMPLATES[chat_template]['description'],
            'prompt': _get_template_with_suffix(chat_template),
        }

    if model_type == "llama3" and model_name_or_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            prompt = _build_tokenizer_prompt(
                tokenizer,
                "{instruction}",
                model_type,
                strip_bos=True,
            )

            template = {
                'description': f"Template from {model_name_or_path} tokenizer",
                'prompt': prompt
            }
            print(f"Using tokenizer template for {model_name_or_path}")
            return template
        except Exception as e:
            print(f"Warning: Could not get Llama 3 tokenizer template: {e}")
            print("Falling back to built-in Llama 3 template")

    if model_type in {"qwen", "gemma"} and model_name_or_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            prompt = _build_tokenizer_prompt(
                tokenizer,
                "{instruction}",
                model_type,
                strip_bos=True,
            )

            template = {
                'description': f"Template from {model_name_or_path} tokenizer",
                'prompt': prompt
            }
            print(f"Using tokenizer template for {model_name_or_path}")
            return template
        except Exception as e:
            print(f"Warning: Could not get {model_type} tokenizer template: {e}")
            if model_type == "qwen":
                print("Falling back to built-in Qwen template")
                return {
                    'description': PROMPT_TEMPLATES["qwen"]['description'],
                    'prompt': _get_template_with_suffix("qwen"),
                }

    type_to_key = {
        "llama2": "llama-2",
        "llama3": "llama-3",
        "mistral": "mistral",
        "qwen": "qwen",
        "vicuna": "vicuna",
    }
    if model_type in type_to_key:
        key = type_to_key[model_type]
        return {
            'description': PROMPT_TEMPLATES[key]['description'],
            'prompt': _get_template_with_suffix(key),
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        messages = [{'role': 'user', 'content': '{instruction}'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt.replace(tokenizer.bos_token, "")

        template = {
            'description': f"Template from {model_name_or_path} tokenizer",
            'prompt': prompt
        }
        print(f"Using tokenizer template for {model_name_or_path}")
        return template
    except Exception as e:
        print(f"Warning: Could not get template from tokenizer: {e}")
        print("Falling back to Llama 2 template")
        return {
            'description': PROMPT_TEMPLATES["llama-2"]['description'],
            'prompt': _get_template_with_suffix("llama-2"),
        }


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: str = 'auto',
    device_map: str = 'auto',
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
    pad_token: str = None,
    eos_token: str = None,
    **kwargs
) -> tuple:
    is_local = os.path.exists(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_STR_DTYPE_TO_TORCH_DTYPE.get(dtype, "auto"),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        local_files_only=is_local,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        local_files_only=is_local,
    )

    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_vllm_model(
    model_name_or_path: str,
    dtype: str = 'float16',
    num_gpus: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.7,
    trust_remote_code: bool = False,
    pad_token: str = None,
    eos_token: str = None,
    **kwargs
) -> LLM:
    if not _VLLM_AVAILABLE:
        raise ModuleNotFoundError(
            "vllm is not installed in this environment. "
            "Install `vllm` or avoid vLLM-backed code paths."
        )

    print(f"Loading vLLM model from {model_name_or_path}...")
    print(f"  dtype={dtype}, num_gpus={num_gpus}, max_len={max_model_len}")

    model = LLM(
        model=model_name_or_path,
        dtype=dtype,
        tensor_parallel_size=num_gpus,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )

    if pad_token:
        model.llm_engine.tokenizer.tokenizer.pad_token = pad_token
    if eos_token:
        model.llm_engine.tokenizer.tokenizer.eos_token = eos_token

    print("Model loaded successfully!")
    return model


def get_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    pad_token: str = None,
    eos_token: str = None,
) -> AutoTokenizer:
    is_local = os.path.exists(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        local_files_only=is_local,
    )

    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
