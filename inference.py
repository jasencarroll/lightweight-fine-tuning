from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import AutoPeftModelForCausalLM
import torch
device = torch.device('cpu')  # Force CPU usage

# Initialize tokenizer with proper settings
# We need to ensure special tokens are handled correctly
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    padding_side="left",  # Important for causal language models
    truncation_side="left",  # Consistent with padding
    use_fast=True  # Use fast tokenizer implementation
)

# Set proper padding token
# GPT-2 doesn't have a pad token by default, so we use EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id




model = AutoModelForCausalLM.from_pretrained("gpt2")
# Set the model to evaluation mode
model.eval()

lora_model = AutoPeftModelForCausalLM.from_pretrained(
    "gpt2-lora",  # Your PEFT model path
    device_map={"": device},  # Force CPU usage
    torch_dtype=torch.float16  # Use float16 for efficiency
)
# Set the model to evaluation mode
lora_model.eval()

# Prepare input text with proper formatting
input_text = "Hello, my name is "  # Your prompt

# Tokenize input with proper parameters
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,  # Adjust based on your needs
    add_special_tokens=True  # Ensure special tokens are added
)
# Generate text with better controlled parameters
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,  # Adjust based on desired output length
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,  # Add nucleus sampling
    temperature=0.7,  # Add temperature for more natural output
    do_sample=True,  # Enable sampling
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2  # Prevent repetitive outputs
)

# Decode the output properly
generated_text = tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True,  # Remove special tokens
    clean_up_tokenization_spaces=True  # Clean up spaces
)

# Generate text with better controlled parameters
outputs = lora_model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,  # Adjust based on desired output length
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,  # Add nucleus sampling
    temperature=0.7,  # Add temperature for more natural output
    do_sample=True,  # Enable sampling
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2  # Prevent repetitive outputs
)

# Decode the output properly
generated_text = tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True,  # Remove special tokens
    clean_up_tokenization_spaces=True  # Clean up spaces
)

print("Generated text:", generated_text[0])