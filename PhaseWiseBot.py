import os
import io
from google.oauth2.service_account import Credentials
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.embeddings.base import Embeddings
import discord
from discord.ext import commands
import openai

# Load sensitive information from environment variables
GOOGLE_API_CREDENTIALS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY

# Check if the credentials file exists
if not GOOGLE_API_CREDENTIALS_FILE or not os.path.exists(GOOGLE_API_CREDENTIALS_FILE):
    raise FileNotFoundError("Google API credentials file not found. Please check the path.")

# Initialize Google Drive credentials
creds = Credentials.from_service_account_file(GOOGLE_API_CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive'])

# Initialize Discord bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Vectorstore global variable
vectorstore = None

class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper to use SentenceTransformer with LangChain."""
    
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        """Embed a single query."""
        return self.model.encode([text])[0]

# Initialize the embedding model
embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

def process_pdfs_to_faiss(creds, folder_id):
    """
    Fetch PDFs from Google Drive, extract text in-memory, process into embeddings,
    and store in a FAISS vector database.
    """
    drive_service = build('drive', 'v3', credentials=creds)

    # Fetch PDFs
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf'",
        fields="files(id, name)"
    ).execute()
    files = results.get('files', [])
    if not files:
        return "No PDF files found in the folder."

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_texts = []

    # Process each PDF in-memory
    for file in files:
        file_id = file['id']
        file_name = file['name']
        print(f"Processing {file_name}...")

        try:
            # Stream the PDF content
            request = drive_service.files().get_media(fileId=file_id)
            pdf_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(pdf_stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            # Extract text from the PDF stream
            pdf_stream.seek(0)  # Reset stream position
            reader = PdfReader(pdf_stream)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            chunks = text_splitter.split_text(text)
            all_texts.extend(chunks)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Generate embeddings and store in FAISS
    if all_texts:
        global vectorstore
        vectorstore = FAISS.from_texts(all_texts, embedding_model)
        vectorstore.save_local("vectorstore_index")
        return f"Processed {len(files)} files and saved embeddings to vectorstore_index."
    else:
        return "No text extracted. Vectorstore not created."

def generate_response(query):
    """
    Generate a chatbot-like response using FAISS and OpenAI's ChatCompletion API.
    """
    if vectorstore is None:
        return "I'm sorry, but my knowledge base is not ready yet. Please use `!update` to initialize the vectorstore."

    # Perform similarity search in FAISS vectorstore
    docs = vectorstore.similarity_search(query, k=5)

    if not docs:
        return "Hmm, I couldn't find anything related to your question. Could you try rephrasing it?"

    # Concatenate the top documents into context for OpenAI
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [
        {"role": "system", "content": "You are a helpful assistant knowledgeable in materials science and engineering named PhaseWise."},
        {"role": "user", "content": f"Here is the context:\n\n{context}\n\nBased on this, please answer the following question:\n{query}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=160,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

@bot.event
async def on_ready():
    """
    Run once when the bot starts.
    """
    print(f"{bot.user} has connected to Discord!")
    print("Updating vectorstore on startup...")
    result = process_pdfs_to_faiss(creds, DRIVE_FOLDER_ID)
    print(result)

@bot.command()
async def update(ctx):
    """
    Command to update the FAISS vectorstore from Google Drive folder.
    """
    await ctx.send("Updating vectorstore... This may take a while.")
    result = process_pdfs_to_faiss(creds, DRIVE_FOLDER_ID)
    await ctx.send(result)

@bot.command()
async def ask(ctx, *, query: str):
    """
    Command to query the FAISS vectorstore with OpenAI-powered responses.
    """
    response = generate_response(query)

    max_length = 2000
    if len(response) > max_length:
        chunks = [response[i:i + max_length] for i in range(0, len(response), max_length)]
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(response)

bot.run(DISCORD_TOKEN)
