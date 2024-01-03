import streamlit as st
import transformers
from transformers import pipeline

# Load the model and tokenizer
model_name = "HuggingFaceH4/zephyr-7b-beta"
model = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = model.tokenizer

# Create the chat interface
st.title("Chat with Zephyr")
st.text("Zephyr is a friendly chatbot who always responds in the style of a pirate.")

# Get the user's message
message = st.text_input("Your message:")

# Generate a response from the model
if message:
    prompt = tokenizer.apply_chat_template([
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": message},
    ], tokenize=False, add_generation_prompt=True)
    outputs = model(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"]

    # Display the response
    st.text(response)
