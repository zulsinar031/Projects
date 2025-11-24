import fitz  # PyMuPDF
import os
from groq import Groq, APIStatusError, APIConnectionError # Import the direct Groq client

# --- Configuration ---
# Ensure your GROQ_API_KEY environment variable is set.
# Example (for temporary setting in terminal before running script):
# export GROQ_API_KEY="your_groq_api_key_here" (Linux/macOS)
# $env:GROQ_API_KEY="your_groq_api_key_here" (PowerShell)

# Initialize Groq client globally
groq_client = Groq()

# Choose a Groq model
# Recommended: "llama3-8b-8192", "mixtral-8x7b-32768", "llama3-70b-8192"
LANGUAGE_MODEL = 'llama3-8b-8192' # Using a common Groq model


# --- PDF Loading and Chunking ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ''
    for page in doc:
        full_text += page.get_text()
    return full_text

# Chunk the text into manageable pieces
def chunk_text(text, max_lines=30):
    lines = text.split('\n')
    chunks = ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]
    return [chunk for chunk in chunks if chunk.strip()]


# --- Groq Integration Function ---

# Generate flashcards from a chunk using Groq
def generate_flashcards_with_groq(chunk_content):
    if not chunk_content.strip():
        return "No text provided for flashcard generation in this chunk."

    # Construct the full prompt content for the user message
    prompt_content = f"""Read the following text and generate up to 3 high-quality flashcards in the form of question-answer pairs. Use clear and concise language.

Format:
Q: [question]
A: [answer]

Text:
{chunk_content}
"""

    try:
        # Call Groq API
        stream = groq_client.chat.completions.create(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': "You are a helpful assistant for generating flashcards."},
                {'role': 'user', 'content': prompt_content}, # The full request including text goes here
            ],
            temperature=0.5, # Moderate temperature for creativity but still factual
            max_tokens=256,  # Adjust token limit based on expected flashcard length
            stream=True,     # Keep stream=True to print tokens as they arrive
        )
        
        flashcards_output = ''
        for part in stream:
            if part.choices and part.choices[0].delta.content is not None:
                flashcards_output += part.choices[0].delta.content
                print(part.choices[0].delta.content, end='', flush=True) # Print immediately
        return flashcards_output.strip() # Return the full generated text
    
    except (APIStatusError, APIConnectionError) as e:
        print(f"\nError calling Groq API for flashcard generation: {e}")
        return f"[Error generating flashcards: {e}]"
    except Exception as e:
        print(f"\nAn unexpected error occurred during flashcard generation: {e}")
        return f"[Unexpected error generating flashcards: {e}]"


# --- Main Execution ---
if __name__ == '__main__':
    pdf_path = 'document.pdf' # Ensure this PDF file exists in the script's directory

    # Check if API key is set
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it before running the script (e.g., export GROQ_API_KEY='your_api_key' or $env:GROQ_API_KEY='your_api_key').")
        exit()

    try:
        full_text = extract_text_from_pdf(pdf_path)
        print(f'Loaded {len(full_text.splitlines())} lines from PDF.')
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found. Please ensure it's in the correct directory.")
        exit()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        exit()

    chunks = chunk_text(full_text, max_lines=30)
    if not chunks:
        print("No usable text chunks found.")
        exit()

    print(f"\nGenerating flashcards from {len(chunks)} chunks...\n")

    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Flashcards from Chunk {i + 1}:\n")
        # Call the Groq-integrated function
        _ = generate_flashcards_with_groq(chunk) # We print directly inside the function
        print("\n" + "-" * 80)