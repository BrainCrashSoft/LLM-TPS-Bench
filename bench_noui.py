import time
import openai

# Configuration
BASE_URL = "http://192.168.178.16:1234/v1"  # Replace with your LM Studio server URL
API_KEY = "lm-studio"
PROMPT = "Explain quantum computing in 1000 words. Include key concepts like superposition, entanglement, and qubits."
MAX_TOKENS = 512
TEMPERATURE = 0.0
RUNS = 1

client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Automatically read model name from LM Studio server
models_response = client.models.list()
available_models = [m.id for m in models_response.data]

if not available_models:
    print("❌ No models found loaded on LM Studio server.")
    exit(1)

MODEL = available_models[0]
print(f"📦 Available model(s): {', '.join(available_models)}")
print(f"🧪 Benchmarking {MODEL}\n")

total_tps = []
for i in range(RUNS):
    start = time.time()
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False
    )
    
    elapsed = time.time() - start
    completion_tokens = response.usage.completion_tokens
    tps = completion_tokens / elapsed
    total_tps.append(tps)
    
    print(f"Run {i+1}: {completion_tokens} tokens | {elapsed:.2f}s | ⚡ {tps:.2f} TPS")

avg_tps = sum(total_tps) / len(total_tps)
print(f"\n✅ Average TPS: {avg_tps:.2f}")
