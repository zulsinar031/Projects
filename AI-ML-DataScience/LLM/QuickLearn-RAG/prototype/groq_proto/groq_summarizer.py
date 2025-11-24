import fitz  # PyMuPDF
import os
from groq import Groq, APIStatusError, APIConnectionError

# --- Configuration ---
groq_client = Groq()
LANGUAGE_MODEL = 'llama3-8b-8192' # Or 'mixtral-8x7b-32768'

# --- PDF Loading and Chunking ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ''
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, max_lines=30):
    lines = text.split('\n')
    chunks = ['\n'.join(lines[i:i+max_lines]) for i in range(0, len(lines), max_lines)]
    return [chunk for chunk in chunks if chunk.strip()]


# --- Groq Integration Functions ---

def summarize_chunk_with_groq(chunk_content):
    # Debug: Print the chunk content before sending
    # print(f"\n--- Chunk Content (first 100 chars): ---\n{chunk_content[:100]}...\n--- End Chunk ---")
    if not chunk_content.strip():
        # print("Warning: Empty chunk content received by summarize_chunk_with_groq.")
        return "[Empty Chunk]" # Return a clear message for empty chunks

    # Simplified prompt for better clarity for the LLM
    prompt_content = f"Please summarize the following text clearly and concisely:\n\n{chunk_content}"

    try:
        chat_completion = groq_client.chat.completions.create(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': "You are a helpful AI tutor for generating summaries."},
                {'role': 'user', 'content': prompt_content}, # Send the whole prompt as user content
            ],
            temperature=0.4,
            max_tokens=300, # Slightly increased max_tokens for testing
            stream=False,
        )
        summary = chat_completion.choices[0].message.content.strip()
        # Debug: Print the summary received
        # print(f"--- Summary for chunk: ---\n{summary}\n--- End Summary ---")
        return summary
    except (APIStatusError, APIConnectionError) as e:
        print(f"Error calling Groq API for chunk summarization: {e}")
        return f"[Error summarizing chunk: {e}]"
    except Exception as e:
        print(f"An unexpected error occurred during chunk summarization: {e}")
        return f"[Unexpected error summarizing chunk: {e}]"


def get_final_summary_with_groq(combined_summaries_input):
    if not combined_summaries_input.strip():
        print("No valid partial summaries were provided for final summarization.")
        return

    # Simplified prompt for final summary
    final_prompt_content = f"Please provide a single, clear, concise overall summary based on these segments:\n\n{combined_summaries_input}"

    try:
        final_chat_completion = groq_client.chat.completions.create(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': "You are a helpful AI assistant for generating overall summaries."},
                {'role': 'user', 'content': final_prompt_content},
            ],
            temperature=0.4,
            max_tokens=600, # Slightly increased max_tokens for testing
            stream=True,
        )
        for chunk_data in final_chat_completion:
            if chunk_data.choices and chunk_data.choices[0].delta.content is not None:
                print(chunk_data.choices[0].delta.content, end='', flush=True)
    except (APIStatusError, APIConnectionError) as e:
        print(f"\nError calling Groq API for final summary: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during final summary: {e}")


# --- Main Execution ---
if __name__ == '__main__':
    pdf_path = 'document.pdf'

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
        print("No usable text chunks found for summarization.")
        exit()

    print(f"Summarizing {len(chunks)} chunks...\n")
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        summary = summarize_chunk_with_groq(chunk)
        partial_summaries.append(summary)
        # Optional: Print each partial summary for debugging
        # print(f"Partial Summary {i+1}: {summary[:50]}...") # Print first 50 chars of summary

    valid_partial_summaries = [s for s in partial_summaries if not s.startswith("[Error summarizing chunk:") and s.strip() != "[Empty Chunk]"]

    if not valid_partial_summaries:
        print("No valid partial summaries were generated. Cannot create a final summary.")
        exit()

    combined_summary_input = '\n'.join(f'- {s}' for s in valid_partial_summaries)

    print("\nFinal Summary:\n")
    get_final_summary_with_groq(combined_summary_input)