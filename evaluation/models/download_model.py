#!/usr/bin/env python3
"""
모델 다운로드 스크립트
사용법: python download_model.py --model llama3-8b-instruct
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download, login
from transformers import AutoModelForCausalLM, AutoTokenizer


# 실험에서 사용하는 모델 목록
MODELS = {
    # Target models
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
    "llama3-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma2-9b-it": "google/gemma-2-9b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3-12b-it": "google/gemma-3-12b-it",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",

    # Vicuna models
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "vicuna-13b": "lmsys/vicuna-13b-v1.5",

    # Classifier
    "harmbench-cls": "cais/HarmBench-Llama-2-13b-cls",

    # Qwen models
    "qwen-7b-chat": "Qwen/Qwen-7B-Chat",
    "qwen-14b-chat": "Qwen/Qwen-14B-Chat",
    "qwen-72b-chat": "Qwen/Qwen-72B-Chat",
    "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",

    # Base models (optional)
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3-8b": "meta-llama/Llama-3.1-8B",
}

# 다운로드 후 저장 디렉토리명 매핑
SAVE_NAMES = {
    "llama2-7b-chat": "llama_2_7b_chat_hf",
    "llama2-13b-chat": "llama_2_13b_chat_hf",
    "llama2-70b-chat": "llama_2_70b_chat_hf",
    "llama3-8b-instruct": "llama-3.1-8b-instruct",
    "llama3-70b-instruct": "llama-3.1-70b-instruct",
    "llama3-3b-instruct": "llama-3.2-3b-instruct",
    "gemma2-9b-it": "gemma-2-9b-it",
    "gemma3_12b": "gemma-3-12b-it",
    "gemma3-12b-it": "gemma-3-12b-it",
    "mistral-7b-instruct": "mistral-7b-instruct-v0.2",
    "harmbench-cls": "harmbench_llama_2_13b_cls",
    # Vicuna models
    "vicuna-7b": "vicuna-7b-v1.5",
    "vicuna-13b": "vicuna-13b-v1.5",
    # Qwen models
    "qwen-7b-chat": "qwen-7b-chat",
    "qwen-14b-chat": "qwen-14b-chat",
    "qwen-72b-chat": "qwen-72b-chat",
    "qwen2-7b-instruct": "qwen2-7b-instruct",
    "qwen2-72b-instruct": "qwen2-72b-instruct",
    "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct": "qwen2.5-14b-instruct",
    "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
}


def get_models_dir() -> Path:
    """models 디렉토리 경로 반환"""
    return Path(__file__).parent


def download_model(
    model_id: str,
    save_dir: Path,
    token: str = None,
) -> str:
    """모델 다운로드 (원본 dtype 그대로 저장, bfloat16 모델은 bfloat16으로 유지)"""
    print(f"다운로드 중: {model_id}")
    print(f"저장 경로: {save_dir}")

    # Tokenizer 다운로드
    print("토크나이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_dir)

    # Model 다운로드 (원본 dtype 유지)
    print("모델 다운로드 중 (원본 dtype 유지)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(save_dir)

    print(f"완료: {save_dir}")
    return str(save_dir)


def list_models():
    """사용 가능한 모델 목록 출력"""
    print("\n사용 가능한 모델:")
    print("-" * 60)
    print(f"{'별칭':<25} {'HuggingFace ID'}")
    print("-" * 60)
    for alias, model_id in MODELS.items():
        print(f"  {alias:<23} {model_id}")
    print("-" * 60)
    print("\n예시:")
    print("  python download_model.py --model llama2-7b-chat")
    print("  python download_model.py --model gemma3_12b")
    print("  python download_model.py --model harmbench-cls")
    print()


def main():
    parser = argparse.ArgumentParser(description="모델 다운로드")

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="다운로드할 모델 (별칭 또는 HuggingFace ID)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="저장할 디렉토리명"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace 토큰 (환경변수 HF_TOKEN도 사용 가능)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="사용 가능한 모델 목록"
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="HuggingFace 로그인"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.login:
        login()
        return

    if not args.model:
        parser.print_help()
        list_models()
        return

    # 모델 ID 확인
    model_id = MODELS.get(args.model, args.model)

    # 토큰
    token = args.token or os.environ.get("HF_TOKEN")

    # 저장 경로
    if args.output:
        save_name = args.output
    else:
        save_name = SAVE_NAMES.get(args.model, args.model.split("/")[-1].lower().replace("-", "_"))

    save_dir = get_models_dir() / save_name
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        download_model(model_id, save_dir, token)
        print(f"\n다운로드 완료: {save_dir}")

    except Exception as e:
        print(f"\n오류: {e}")
        print("\n해결 방법:")
        print("1. HF_TOKEN 환경변수 설정 또는 --token 옵션 사용")
        print("2. HuggingFace에서 모델 라이선스 동의")
        print("3. python download_model.py --login 으로 로그인")
        raise


if __name__ == "__main__":
    main()
