from llm_sdk import Small_LLM_Model
import torch

def main():
    try:
        model = Small_LLM_Model()
        
        # 1. Test Encoding
        test_text = "What is 2 + 3?"
        token_ids = model.encode(test_text)
        print(f"Raw Output from encode: {token_ids}")

        # 2. CONVERSION STEP: 
        # The SDK returns a tensor [[ids]]. We need a list [ids].
        if torch.is_tensor(token_ids):
            # Move to CPU, convert to list, and take the first row
            token_list = token_ids.cpu().tolist()[0]
        else:
            token_list = token_ids

        # 3. Now pass the LIST to get_logits
        logits = model.get_logits_from_input_ids(token_list)
        
        print(f"Logits length: {len(logits)}")
        print("✅ SDK verification successful!")

    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    main()