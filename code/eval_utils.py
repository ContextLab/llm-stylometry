import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, eval_dataloader, device):
    """
    Returns average loss across the evaluation dataset batches
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            # Use mixed precision for evaluation too
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                input_ids = batch["input_ids"].to(device)
                # Only use attention_mask if it exists
                if "attention_mask" in batch:
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
                    )
                else:
                    outputs = model(input_ids=input_ids, labels=input_ids)

                loss = outputs.loss.item()
                total_loss += loss

            # Clean up memory after each batch
            del outputs
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

    return total_loss / len(eval_dataloader)


# def generate_sample_text(model_name, prompt="once upon a time", max_length=500, tokenizer_name="gpt2") -> str:
#     from transformers import GPT2LMHeadModel
#
#     model_dir = MODELS_DIR / model_name
#     tokenizer = get_tokenizer(tokenizer_name)
#
#     model = GPT2LMHeadModel.from_pretrained(model_dir)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#
#     encoding = tokenizer(prompt, return_tensors="pt")
#     input_ids = encoding["input_ids"].to(device)
#     attention_mask = encoding["attention_mask"].to(device)
#
#     output_ids = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_length=max_length,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#
#     sample_path = f"{model_dir}/{model_name}_sample.txt"
#     with open(sample_path, "w", encoding="utf-8") as f:
#         f.write(generated_text)
#     print(f"Sample generated for {model_name} and saved to {sample_path}")


# if __name__ == "__main__":
#     for model_name in ["baum_gpt_seed=1", "thompson_gpt_seed=1"]:
#         generate_sample_text(model_name, tokenizer_name="gpt2")
