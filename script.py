import requests
import json
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:0.5b"


def send_request(prompt: str) -> str:
    """
    Отправляет запрос к Ollama серверу и возвращает ответ модели.

    Args:
        prompt (str): Текст запроса к языковой модели.

    Returns:
        str: Текстовый ответ от модели.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def run_inference(prompts: list) -> list:
    """
    Прогоняет список запросов через модель и собирает результаты.

    Args:
        prompts (list): Список строк-запросов.

    Returns:
        list: Список словарей с полями 'prompt' и 'response'.
    """
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Запрос: {prompt[:50]}...")
        response = send_request(prompt)
        results.append({"prompt": prompt, "response": response})
        print(f"  Ответ: {response[:80]}...\n")
    return results


def save_report(results: list, filename: str = "report.csv") -> None:
    """
    Сохраняет отчёт инференса в CSV файл с двумя столбцами.

    Args:
        results (list): Список словарей с полями 'prompt' и 'response'.
        filename (str): Имя выходного CSV файла.
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Отчёт сохранён в {filename}")


PROMPTS = [
    "What is the capital of France?",
    "Explain what machine learning is in one sentence.",
    "Write a haiku about winter.",
    "What is 17 multiplied by 13?",
    "Name three programming languages.",
    "What does HTTP stand for?",
    "Describe the water cycle briefly.",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Give one tip for better sleep.",
]

if __name__ == "__main__":
    results = run_inference(PROMPTS)
    save_report(results)