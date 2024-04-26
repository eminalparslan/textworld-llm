from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer
import transformers
import torch

model = "./submission_llm/llama-2-7b/output/"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

print(len(sequences))
print(sequences)
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
