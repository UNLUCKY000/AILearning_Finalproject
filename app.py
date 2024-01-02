import streamlit as st
from transformers import pipeline
import torch

st.title("Pirate story generator")

# Create a text input field for the user to enter a prompt
prompt = st.text_input("Enter a prompt for the story:")

# Create a button for the user to generate the story
generate_button = st.button("Generate story")

# Create a text output field to display the generated story
# story = st.empty()

# Create a function to generate the story
def generate_story(prompt):
  # Load the tokenizer and model
  pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
  # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
  messages = [
      {
          "role": "system",
          "content": "You are a friendly chatbot who always responds in the style of a pirate",
      },
      {"role": "user", "content": prompt},
  ]
  prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
  return outputs[0]["generated_text"]

# Check if the generate button was clicked
if generate_button:
  # Generate the story
  st.write(generate_story(prompt))
