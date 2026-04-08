import asyncio
import os
import sys
import time
from pathlib import Path

# Add client directory to path to ensure we can import anything from there if needed
client_dir = Path(__file__).resolve().parent / "client"
sys.path.insert(0, str(client_dir))

try:
    import tinker
except ImportError:
    print("Could not import tinker. Please make sure you run this using the client's venv:")
    print("  client/.venv/bin/python test_api_support.py")
    sys.exit(1)

async def test_method(client_name, method_name, coro, timeout=5):
    print(f"Testing {client_name}.{method_name}...")
    try:
        start = time.time()
        result = await asyncio.wait_for(coro, timeout=timeout)
        print(f"  ✅ {client_name}.{method_name}: Success (took {time.time() - start:.2f}s)")
        return True, result
    except asyncio.TimeoutError:
        print(f"  ❌ {client_name}.{method_name}: Timed out after {timeout}s (Likely NOT supported by server)")
        return False, "Timeout"
    except Exception as e:
        print(f"  ❌ {client_name}.{method_name}: Failed with error: {e}")
        return False, str(e)

async def main():
    base_url = os.getenv("TINKER_BASE_URL", "http://localhost:8000")
    print(f"Connecting to ServiceClient at {base_url}\n")
    client = tinker.ServiceClient(base_url=base_url)
    
    # ---------------------------------------------------------
    # 1. ServiceClient Methods
    # ---------------------------------------------------------
    print("=== Testing ServiceClient ===")
    service_methods = [
        'get_server_capabilities_async', 'get_telemetry', 
        'create_lora_training_client_async', 'create_sampling_client',
        'create_training_client_from_state_async', 
        'create_training_client_from_state_with_optimizer_async'
    ]
    print(f"Known methods to test: {service_methods}\n")
    
    await test_method("ServiceClient", "get_server_capabilities_async", client.get_server_capabilities_async())
    
    try:
        client.get_telemetry()
        print("  ✅ ServiceClient.get_telemetry: Success")
    except Exception as e:
        print(f"  ❌ ServiceClient.get_telemetry: Failed: {e}")
        
    # Get default model for testing
    default_model = "google/gemma-2b"
    try:
        caps = await client.get_server_capabilities_async()
        if caps.get("default_model"):
            default_model = caps["default_model"]
    except:
        pass
        
    print(f"\nCreating TrainingClient with model: {default_model}")
    success, trainer = await test_method(
        "ServiceClient", "create_lora_training_client_async", 
        client.create_lora_training_client_async(base_model=default_model, rank=8)
    )
    
    # ---------------------------------------------------------
    # 2. TrainingClient Methods
    # ---------------------------------------------------------
    if success and trainer:
        print("\n=== Testing TrainingClient ===")
        trainer_methods = [
            'get_info_async', 'get_tokenizer', 'get_telemetry',
            'forward_async', 'forward_backward_async', 'optim_step_async',
            'save_state_async', 'save_weights_for_sampler_async',
            'save_weights_and_get_sampling_client_async', 'load_state_async'
        ]
        print(f"Known methods to test: {trainer_methods}\n")
        
        await test_method("TrainingClient", "get_info_async", trainer.get_info_async())
        
        try:
            tok = trainer.get_tokenizer()
            print("  ✅ TrainingClient.get_tokenizer: Success")
        except Exception as e:
            print(f"  ❌ TrainingClient.get_tokenizer: Failed: {e}")
            
        try:
            trainer.get_telemetry()
            print("  ✅ TrainingClient.get_telemetry: Success")
        except Exception as e:
            print(f"  ❌ TrainingClient.get_telemetry: Failed: {e}")
            
        # Test save_weights_for_sampler (expect timeout if not supported)
        await test_method("TrainingClient", "save_weights_for_sampler_async", trainer.save_weights_for_sampler_async(name="probe_test_weights"))
        
        # Test save_state (expect timeout)
        await test_method("TrainingClient", "save_state_async", trainer.save_state_async(name="probe_test_state"))

    # ---------------------------------------------------------
    # 3. SamplingClient Methods
    # ---------------------------------------------------------
    print("\n=== Testing SamplingClient ===")
    print("Creating SamplingClient with dummy path...")
    try:
        sampler = client.create_sampling_client("tinker://dummy|/tmp/dummy")
        print("  ✅ ServiceClient.create_sampling_client: Success (Handle created)")
        
        sampler_methods = ['sample_async', 'compute_logprobs_async', 'get_tokenizer', 'get_telemetry']
        print(f"Known methods to test: {sampler_methods}\n")
        
        try:
            sampler.get_telemetry()
            print("  ✅ SamplingClient.get_telemetry: Success")
        except Exception as e:
            print(f"  ❌ SamplingClient.get_telemetry: Failed: {e}")
            
        # Test sample_async with dummy input (expect failure or timeout because path is dummy)
        # But we want to see if it reaches the server
        await test_method(
            "SamplingClient", "sample_async", 
            sampler.sample_async(
                prompt=tinker.types.ModelInput.from_ints(tokens=[1, 2, 3]),
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(max_tokens=5)
            ),
            timeout=5
        )
        
    except Exception as e:
        print(f"  ❌ Failed to create SamplingClient handle: {e}")

if __name__ == "__main__":
    asyncio.run(main())
