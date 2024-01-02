import streamlit as st
import torch
from transformers import pipeline

def generate_pirate_story():
    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    # System and user messages
    messages = [
        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
        {"role": "user", "content": st.text_input("Write a story", "Write a story")},
    ]

    # Apply chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate text
    outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    # Display generated text
    return outputs[0]["generated_text"]

def main():
    st.title("Pirate Story Generator")
    
    # Generate and display pirate story
    generated_text = generate_pirate_story()
    st.text_area("Generated Pirate Story", generated_text, height=300)

if __name__ == "__main__":
    main()
