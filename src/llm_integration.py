import os
import re
import json
import ollama
import logging
from config import MODEL, TEMPERATURE, MODEL_API_URL, PROMPTS_CONFIG_FILE, VERBOSE
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

global llm_communication
llm_communication = False  # Global variable to track LLM connectivity


def check_llm_connectivity():
    global llm_communication
    try:
        client = ollama.Client(host=MODEL_API_URL)
        response = client.show(MODEL)
        if response:
            print("Ollama LLM is reachable.")
            llm_communication = True
        else:
            print("Ollama LLM is not reachable.")
            llm_communication = False
    except Exception as e:
        print(f"Failed to reach Ollama LLM: {str(e)}")
        llm_communication = False

class PromptProcessor:
    def __init__(self, output_folder, custom_hooks=None):
        self.model = MODEL
        self.temperature = TEMPERATURE
        self.output_folder = output_folder
        self.custom_hooks = custom_hooks if custom_hooks else []
        self.client = ollama.Client(host=MODEL_API_URL)

    def load_prompts_from_json(self, json_file):
        """Load prompts from a JSON file."""
        try:
            if VERBOSE:
                print("Loading prompts from JSON configuration...")
            with open(json_file, 'r', encoding='utf-8') as file:
                config = json.load(file)
                self.messages = config['messages']
        except (OSError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load prompts from {json_file}: {str(e)}")

    def process_messages(self, content, file_name):
        """Process the content, preserving code blocks."""
        # Step 1: Identify code blocks and replace them with placeholders
        code_blocks = re.findall(r'```.*?```', content, re.DOTALL)
        placeholders = {f"[[CODE_BLOCK_{i}]]": code_block for i, code_block in enumerate(code_blocks)}
        content_with_placeholders = content
        for placeholder, code_block in placeholders.items():
            content_with_placeholders = content_with_placeholders.replace(code_block, placeholder)

        # Step 2: Format the messages for LLM processing
        messages = self._format_messages(content_with_placeholders)

        try:
            response = self.client.chat(model=self.model, messages=messages)
            final_response = self._get_response_content(response)

            if not final_response:
                logging.error(f"No response generated for {file_name}.")
                return ""

            # Step 3: Replace placeholders with the original code blocks
            for placeholder, code_block in placeholders.items():
                final_response = final_response.replace(placeholder, code_block)

            for hook in self.custom_hooks:
                final_response = hook(final_response)

            self._save_response(file_name, final_response)
            return final_response
        except ollama.ResponseError as e:
            logging.error(f"Failed to process messages for {file_name}: {e}")
            return ""

    def _format_messages(self, content):
        """Format messages with the provided content."""
        return [
            {
                "role": message['role'],
                "content": "\n".join([part['text'] for part in message['content']]).format(content=content)
            }
            for message in self.messages
        ]

    def _get_response_content(self, response):
        """Extract content from the model's response."""
        final_response = ""

        try:
            if isinstance(response, list):
                for part in response:
                    if isinstance(part, dict) and 'message' in part:
                        message_content = part['message'].get('content', '')
                        final_response += message_content
                        if VERBOSE and message_content:
                            print(message_content, end='', flush=True)
                    else:
                        logging.error(f"Unexpected response format: {part}")
            elif isinstance(response, dict):
                if 'message' in response:
                    final_response = response['message'].get('content', '')
                    if VERBOSE and final_response:
                        print(final_response, end='', flush=True)
                else:
                    logging.error(f"Unexpected response structure: {response}")
            else:
                logging.error(f"Unexpected response type: {type(response)}")

            if not final_response:
                logging.error("The final response content is empty.")

        except TypeError as e:
            logging.error(f"TypeError encountered while processing response: {e}")

        return final_response

    def _save_response(self, file_name, content):
        """Save the model's response to a file."""
        output_path = os.path.join(self.output_folder, file_name)
        output_path = os.path.splitext(output_path)[0] + ".md"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not content:
            logging.error(f"No content to save for {file_name}.")
            return

        try:
            logging.info(f"Saving content to {output_path}...")
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Content successfully saved to {output_path}.")
            if VERBOSE:
                print("\nFinal Content:\n", content)
        except OSError as e:
            logging.error(f"Failed to save response for {file_name}: {str(e)}")

    def run(self, initial_content, file_name):
        """Run the processor with the initial content and file name."""
        if VERBOSE:
            print(f"\n********************* Thinking...\n")
        final_content = self.process_messages(initial_content, file_name)
        if VERBOSE:
            print(f"Generated content for {file_name}:\n{final_content}")
        if not final_content:
            logging.warning(f"No content was generated for {file_name}.")
        return final_content

def process_with_langchain(rag_path):
    loader = TextLoader(rag_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()  # Replace with your actual embedding model
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = OpenAI()  # Replace with your actual LLM
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        result = qa_chain({"query": query})
        print(result['output'])