from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain # While LLMChain can be used with custom LLMs, it's more common with LangChain's own LLM classes.
                                     # For direct Groq client, we'll make a slight adjustment.
import random
import os
from groq import Groq, APIStatusError, APIConnectionError # Import the direct Groq client

# 1. Load PDF and split into chunks
def load_chunks_from_pdf(pdf_path, chunk_size=1000, chunk_overlap=100):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# 2. Define prompt template for MCQ generation
# This prompt template is designed for direct LLM interaction, less specific to LangChain's LLMChain input format
mcq_prompt_template_str = """
You are a helpful assistant. Based on the following text, generate one multiple-choice question.

Requirements:
- A clear and meaningful question.
- Four answer choices labeled A., B., C., D.
- Clearly indicate the correct answer at the end using this format: Correct Answer: <A/B/C/D>

Text:
\"\"\"{text}\"\"\"

Output format:
Question: <your question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Correct Answer: <A/B/C/D>
"""

# 3. Create a function to interact with the Groq API directly
# We won't use LLMChain in the same way, as we're now directly calling the Groq client.
def generate_mcq_with_groq(text_content, model_name="llama3-8b-8192", temperature=0.7):
    # Ensure GROQ_API_KEY is set in environment or pass it directly
    groq_client = Groq() # Initializes with API key from env by default

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI tutor for generating multiple-choice questions."
                },
                {
                    "role": "user",
                    "content": mcq_prompt_template_str.format(text=text_content)
                }
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=512, # Adjust max_tokens as needed for MCQ length
        )
        return chat_completion.choices[0].message.content
    except (APIStatusError, APIConnectionError) as e:
        print(f"Error calling Groq API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# 4. Generate MCQs (modified to use the direct Groq function)
def generate_mcqs_from_chunks(chunks, max_mcqs=10, model_name="llama3-8b-8192"):
    selected_chunks = random.sample(chunks, min(max_mcqs, len(chunks)))
    print(f"\nGenerating MCQs from {len(selected_chunks)} chunks using direct Groq API...\n")
    mcqs = []
    for i, chunk in enumerate(selected_chunks, 1):
        print(f"Processing chunk {i}/{len(selected_chunks)}...")
        result = generate_mcq_with_groq(chunk.page_content, model_name=model_name)
        if result:
            mcqs.append(result.strip())
            print(f"\nMCQ {i}:\n{result.strip()}")
        else:
            print(f"Skipping MCQ {i} due to an error during generation.")
    return mcqs

# 5. Run everything
if __name__ == '__main__':
    pdf_path = "document.pdf" # Make sure this file exists in the same directory
    groq_model_name = "llama3-8b-8192" # Or "mixtral-8x7b-32768", etc.

    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it before running the script (e.g., export GROQ_API_KEY='your_api_key').")
    else:
        try:
            chunks = load_chunks_from_pdf(pdf_path)
            print(f'Loaded {len(chunks)} chunks from PDF.')

            # The LLMChain abstraction is less direct here, as we're managing the API call ourselves.
            # We're passing the chunks directly to the new generation function.
            generate_mcqs_from_chunks(chunks, model_name=groq_model_name)

        except FileNotFoundError:
            print(f"Error: The file '{pdf_path}' was not found. Please ensure it's in the correct directory.")
        except Exception as e:
            print(f"An error occurred: {e}")