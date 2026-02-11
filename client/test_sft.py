import os
import asyncio
import numpy as np
import tinker
from tinker import types
import matplotlib.pyplot as plt

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

async def main():
    print("1. Initializing Service Client...")
    service_client = tinker.ServiceClient()

    base_model = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"\n3. Creating LoRA Training Client for '{base_model}'...")
    try:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model, rank=16
        )
    except Exception as e:
        print(f"Error creating client: {e}")
        return

    tokenizer = training_client.get_tokenizer()

    questions = [ "What's the weather like today?", "How do I reset my password?", "Can you explain quantum computing?", "What are the health benefits of exercise?", "How do I bake chocolate chip cookies?", "What's the capital of France?", "How does photosynthesis work?", "What are some good books to read?", "How can I learn Python programming?", "What causes climate change?", "How do I fix a leaky faucet?", "What's the difference between AI and machine learning?", "How do I write a resume?", "What are the symptoms of the flu?", "How does blockchain technology work?", "What's the best way to save money?", "How do I meditate effectively?", "What are the planets in our solar system?", "How can I improve my sleep quality?", "What's the history of the Internet?" ]

    dataset = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "foo"}]} for q in questions]

    def make_datum(example):
        # Full conversation tokens
        text_tokens = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)
        tokens = tokenizer.encode(text_tokens, add_special_tokens=False)
        # Prompt-only tokens (to find where completion starts)
        text_prompt = tokenizer.apply_chat_template(example["messages"][:-1], add_generation_prompt=True, tokenize=False)
        prompt_tokens = tokenizer.encode(text_prompt, add_special_tokens=False)
        # Weights: 0 for prompt, 1 for completion
        weights = [0] * len(prompt_tokens) + [1] * (len(tokens) - len(prompt_tokens))
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = weights[1:]  # Shift to align with targets
    
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
        )

    datums = [make_datum(ex) for ex in dataset]
    print(f"Generated {len(datums)} datums.")

    history = []
    print("\nTraining Loop:")
    for epoch in range(20):
        fwdbwd_future = await training_client.forward_backward_async(datums, "cross_entropy")
        fwdbwd_result = await fwdbwd_future

        optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=5e-4))
        optim_result = await optim_future

        # Compute loss
        logprobs = np.concatenate([out['logprobs'].tolist() for out in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs['weights'].tolist() for d in datums])
        loss = -np.dot(logprobs, weights) / weights.sum()
        print(f"Epoch {epoch+1}: loss = {loss:.4f}")
        history.append(loss)

    print("\n-> Generating loss curve plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker='o', linestyle='-', color='b')
    plt.title('SFT Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sft_loss.png')
    print("Saved 'sft_loss.png'")

    sampling_client = training_client.save_weights_and_get_sampling_client(name="foo_v1")
    test_messages = [{"role": "user", "content": "What's your favorite color?"}]
    test_text = tokenizer.apply_chat_template(test_messages, add_generation_prompt=True, tokenize=False)
    test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
    response = sampling_client.sample(
        prompt=types.ModelInput.from_ints(tokens=test_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=20, temperature=0.7)
    ).result()
    print(type(response))
    print(response)
    print(tokenizer.decode(response.sequences[0].tokens))

if __name__ == "__main__":
    asyncio.run(main())
