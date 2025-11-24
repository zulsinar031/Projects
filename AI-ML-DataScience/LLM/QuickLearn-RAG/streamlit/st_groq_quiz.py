import streamlit as st
import os
import tempfile
import random
import re

# Langchain and Groq imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq, APIStatusError, APIConnectionError

# --- Configuration ---
#GROQ_MODEL_NAME = "llama3-8b-8192" # Or "mixtral-8x7b-32768", etc.
GROQ_MODEL_NAME = "llama3-70b-8192"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_MCQS = 3 # Limit the number of MCQs to generate for demonstration/performance

# --- Prompt Template for MCQ Generation ---
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

# --- Prompt Template for Quiz Feedback ---
feedback_prompt_template_str = """
You are an AI tutor providing feedback on a multiple-choice quiz.

Here are the quiz results:
{results_summary}

Based on these results, provide constructive feedback. Consider:
- An encouraging overall summary.
- Identifying topics or questions where the user might need more review (especially for incorrect answers).
- General study tips or next steps.

Keep the feedback concise and helpful.
"""

# --- Functions ---

@st.cache_data(show_spinner="Loading and splitting PDF into chunks...")
def load_chunks_from_pdf(pdf_file_path, chunk_size, chunk_overlap):
    """
    Loads a PDF document from a given path and splits it into text chunks.
    This function is cached by Streamlit to avoid re-processing the PDF
    if the input file doesn't change.
    """
    try:
        loader = PyMuPDFLoader(pdf_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        st.success(f"Loaded {len(chunks)} chunks from the PDF.")
        return chunks
    except Exception as e:
        st.error(f"Error loading or splitting PDF: {e}")
        return []

def generate_mcq_with_groq(text_content, groq_api_key, model_name=GROQ_MODEL_NAME, temperature=0.7):
    """
    Interacts directly with the Groq API to generate a single MCQ.
    """
    if not groq_api_key:
        # This error is handled higher up, but kept here for function independence
        return None

    try:
        groq_client = Groq(api_key=groq_api_key)
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
            max_tokens=512,
        )
        return chat_completion.choices[0].message.content
    except (APIStatusError, APIConnectionError) as e:
        st.error(f"Groq API Error during MCQ generation: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during MCQ generation: {e}")
        return None

def generate_feedback_with_groq(results_summary, groq_api_key, model_name=GROQ_MODEL_NAME, temperature=0.5):
    """
    Generates quiz feedback using the Groq API based on the quiz results summary.
    """
    if not groq_api_key:
        return "Cannot generate feedback: Groq API Key is not set."

    try:
        groq_client = Groq(api_key=groq_api_key)
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an encouraging and insightful AI tutor providing quiz feedback."
                },
                {
                    "role": "user",
                    "content": feedback_prompt_template_str.format(results_summary=results_summary)
                }
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=500, # Adjust token limit for feedback
        )
        return chat_completion.choices[0].message.content
    except (APIStatusError, APIConnectionError) as e:
        return f"Error generating feedback from Groq API: {e}"
    except Exception as e:
        return f"An unexpected error occurred during feedback generation: {e}"

def parse_mcq_output(mcq_text):
    """
    Parses the raw text output from the LLM into a structured dictionary.
    """
    if not mcq_text:
        return None

    try:
        question_match = re.search(r"Question: (.*?)\n[A-D]\.", mcq_text, re.DOTALL)
        if question_match:
            question = question_match.group(1).strip()
        else:
            # Fallback if regex doesn't capture it perfectly
            question_line_start = mcq_text.find("Question:")
            if question_line_start != -1:
                lines = mcq_text[question_line_start + len("Question:"):].split('\n')
                question_parts = []
                for line in lines:
                    if re.match(r"^[A-D]\.", line.strip()):
                        break
                    if line.strip():
                        question_parts.append(line.strip())
                question = " ".join(question_parts) if question_parts else "Error parsing question."
            else:
                question = "Error parsing question: 'Question:' prefix not found."

        options = {}
        option_a_match = re.search(r"A\. (.+)", mcq_text)
        option_b_match = re.search(r"B\. (.+)", mcq_text)
        option_c_match = re.search(r"C\. (.+)", mcq_text)
        option_d_match = re.search(r"D\. (.+)", mcq_text)

        if option_a_match: options['A'] = option_a_match.group(1).strip()
        if option_b_match: options['B'] = option_b_match.group(1).strip()
        if option_c_match: options['C'] = option_c_match.group(1).strip()
        if option_d_match: options['D'] = option_d_match.group(1).strip()

        for k in ['A', 'B', 'C', 'D']:
            if k not in options:
                options[k] = f"Option {k} missing."

        correct_answer_match = re.search(r"Correct Answer: ([A-D])", mcq_text)
        correct_answer_key = correct_answer_match.group(1).strip() if correct_answer_match else "N/A"

        return {
            "question": question,
            "options": options,
            "correct_answer_key": correct_answer_key,
            "correct_answer_text": options.get(correct_answer_key, "Not found")
        }
    except Exception as e:
        st.error(f"Failed to parse MCQ text due to an internal error: {e}. Raw text:\n{mcq_text}")
        return None

@st.cache_data(show_spinner="Generating MCQs from selected chunks...")
def generate_mcqs_from_chunks(_chunks, groq_api_key, max_mcqs, model_name):
    """
    Selects random chunks and generates MCQs for them using the Groq API.
    """
    if not _chunks:
        st.warning("No chunks available to generate MCQs from.")
        return []

    selected_chunks = random.sample(_chunks, min(max_mcqs, len(_chunks)))
    mcqs_raw = []
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, chunk in enumerate(selected_chunks):
        progress_text.text(f"Generating MCQ {i+1} of {len(selected_chunks)}...")
        progress_bar.progress((i + 1) / len(selected_chunks))
        result = generate_mcq_with_groq(chunk.page_content, groq_api_key, model_name=model_name)
        if result:
            mcqs_raw.append(result.strip())
        else:
            st.warning(f"Skipping MCQ {i+1} due to an error during generation.")
    progress_text.empty()
    progress_bar.empty()
    return mcqs_raw

# --- Streamlit UI ---
st.set_page_config(page_title="PDF to MCQ Generator", layout="centered")

st.title("📚 PDF to MCQ Generator")
st.markdown(
    """
    Upload a PDF document and let the AI generate multiple-choice questions (MCQs)
    from its content. Powered by Groq for fast LLM inference!
    """
)

# Initialize session state variables
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'upload_and_generate' # 'upload_and_generate' or 'quiz' or 'results'
if 'mcqs' not in st.session_state:
    st.session_state.mcqs = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {} # Stores user's selected answers for each question
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_feedback' not in st.session_state:
    st.session_state.quiz_feedback = None

# --- API Key Input (always visible) ---
groq_api_key = st.text_input(
    "Enter your Groq API Key",
    type="password",
    help="You can get your API key from https://console.groq.com/keys",
    key="groq_api_key_input" # Persistent key for the input widget
)

# --- View Logic ---

if st.session_state.current_view == 'upload_and_generate':
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"], key="pdf_uploader")

    if uploaded_file:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        st.info(f"Processing '{uploaded_file.name}'...")

        # Load and split chunks
        chunks = load_chunks_from_pdf(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP)

        if chunks:
            if st.button(f"Generate {MAX_MCQS} MCQs", key="generate_button"):
                if not groq_api_key:
                    st.error("Please enter your Groq API Key before generating MCQs.")
                else:
                    with st.spinner("Generating questions... This may take a moment."):
                        mcqs_raw_output = generate_mcqs_from_chunks(chunks, groq_api_key, MAX_MCQS, GROQ_MODEL_NAME)

                    parsed_mcqs = [parse_mcq_output(mcq_text) for mcq_text in mcqs_raw_output]
                    st.session_state.mcqs = [mcq for mcq in parsed_mcqs if mcq is not None]
                    st.session_state.user_answers = {i: None for i in range(len(st.session_state.mcqs))}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_feedback = None # Clear previous feedback

                    if not st.session_state.mcqs:
                        st.warning("No valid MCQs could be generated or parsed. Please check the PDF content and API key.")
                    else:
                        st.success(f"Successfully generated {len(st.session_state.mcqs)} MCQs! Ready for quiz.")
                        st.session_state.current_view = 'quiz'
                        st.rerun() # Move to quiz view
        else:
            st.warning("No chunks were processed. Please ensure the PDF is valid and not empty.")

    # Cleanup temporary file. This happens when `pdf_path` is set.
    # Note: Streamlit reruns, so this might trigger multiple times.
    # The `delete=False` ensures it persists for the session if needed.
    # A more robust cleanup might involve st.session_state and os.remove() in a cleanup function.
    if 'pdf_path' in locals() and os.path.exists(pdf_path):
        try:
            # os.remove(pdf_path) # Commented out for easier debugging if file is needed later
            pass
        except OSError as e:
            print(f"Error removing temporary file {pdf_path}: {e}")

elif st.session_state.current_view == 'quiz':
    st.header("Take the Quiz!")
    if not st.session_state.mcqs:
        st.warning("No questions generated yet. Please go back to generate questions.")
        if st.button("Go Back to Generation", key="go_back_from_quiz"):
            st.session_state.current_view = 'upload_and_generate'
            st.rerun()
        st.stop()

    # Display MCQs
    all_answered = True
    for i, mcq in enumerate(st.session_state.mcqs):
        st.subheader(f"Question {i+1}")
        st.markdown(f"**Q: {mcq['question']}**")

        ordered_options = sorted(mcq['options'].items())
        options_list = [f"{key}. {text}" for key, text in ordered_options]
        
        # User's choice for this question
        current_selection = st.session_state.user_answers.get(i)
        
        # Determine the default index for st.radio
        # Find index of current_selection in options_list
        default_index = None
        if current_selection:
            try:
                default_index = options_list.index(f"{current_selection}. {mcq['options'][current_selection]}")
            except ValueError:
                default_index = None # Should not happen if parsing is correct

        user_choice_raw = st.radio(
            f"Select your answer for Question {i+1}:",
            options_list,
            key=f"q{i}_radio",
            index=default_index,
            help="Choose one option."
        )
        
        if user_choice_raw:
            selected_key = user_choice_raw.split('.')[0]
            st.session_state.user_answers[i] = selected_key
        else:
            st.session_state.user_answers[i] = None # Ensure it's explicitly None if nothing selected

        if st.session_state.user_answers[i] is None:
            all_answered = False
        
        st.markdown("---") # Separator

    st.markdown("---") # Another separator before submit button

    # Submit Answers Button
    if st.button("Submit Answers", key="submit_answers_button"):
        # Check if all questions have been answered
        if all_answered:
            st.session_state.quiz_submitted = True
            st.session_state.current_view = 'results'
            st.rerun()
        else:
            st.warning("Please answer all questions before submitting.")

    if st.button("Go Back to Generation", key="go_back_from_quiz_bottom"):
        st.session_state.current_view = 'upload_and_generate'
        st.rerun()

elif st.session_state.current_view == 'results':
    st.header("Quiz Review & Feedback")

    correct_count = 0
    incorrect_answers_summary = []

    # Display Results for Each Question
    for i, mcq in enumerate(st.session_state.mcqs):
        user_answer_key = st.session_state.user_answers.get(i)
        is_correct = (user_answer_key == mcq['correct_answer_key'])

        st.subheader(f"Question {i+1}: {'✅ Correct' if is_correct else '❌ Incorrect'}")
        st.markdown(f"**Q: {mcq['question']}**")

        # Display all options with highlighting
        for option_key, option_text in sorted(mcq['options'].items()):
            display_text = f"{option_key}. {option_text}"
            if option_key == mcq['correct_answer_key']:
                st.markdown(f"<span style='color:green; font-weight:bold;'>{display_text} (Correct Answer)</span>", unsafe_allow_html=True)
            elif option_key == user_answer_key:
                st.markdown(f"<span style='color:red; font-weight:bold;'>{display_text} (Your Answer)</span>", unsafe_allow_html=True)
            else:
                st.write(display_text)
        
        st.markdown("---")

        if not is_correct:
            incorrect_answers_summary.append(
                f"Q{i+1}: {mcq['question']}\n"
                f"Your Answer: {user_answer_key}. {mcq['options'].get(user_answer_key, 'N/A')}\n"
                f"Correct Answer: {mcq['correct_answer_key']}. {mcq['correct_answer_text']}\n"
            )
        else:
            correct_count += 1

    st.markdown("## Overall Performance")
    total_questions = len(st.session_state.mcqs)
    st.write(f"You answered {correct_count} out of {total_questions} questions correctly.")

    # Generate and display feedback if not already done
    if st.session_state.quiz_feedback is None:
        with st.spinner("Generating personalized feedback..."):
            results_string = ""
            for i, mcq in enumerate(st.session_state.mcqs):
                user_ans = st.session_state.user_answers.get(i)
                results_string += f"Question {i+1}: {mcq['question']}\n"
                results_string += f"User's Choice: {user_ans}. {mcq['options'].get(user_ans, 'No answer selected')}\n"
                results_string += f"Correct Answer: {mcq['correct_answer_key']}. {mcq['correct_answer_text']}\n"
                results_string += f"Result: {'Correct' if user_ans == mcq['correct_answer_key'] else 'Incorrect'}\n\n"
            
            # Add incorrect answers details to prompt
            if incorrect_answers_summary:
                results_string += "\nDetails on Incorrect Answers:\n" + "\n".join(incorrect_answers_summary)

            st.session_state.quiz_feedback = generate_feedback_with_groq(results_string, groq_api_key)
            st.rerun() # Rerun to display feedback

    if st.session_state.quiz_feedback:
        st.markdown("### AI Tutor Feedback:")
        st.info(st.session_state.quiz_feedback)

    if st.button("Retake Quiz / Generate New Questions", key="retake_quiz_button"):
        st.session_state.current_view = 'upload_and_generate'
        st.session_state.mcqs = [] # Clear MCQs for new generation
        st.session_state.user_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_feedback = None
        st.rerun()

