# PhaseWise: AI-Powered PDF Knowledge Assistant

PhaseWise is a Discord bot designed to extract, process, and understand content from PDF documents stored in Google Drive. It uses advanced machine learning techniques to build a searchable knowledge base and answer user queries interactively. Phase Wise was created to consolodate the information from my undergraduate Materials Science and Engineering courses at The Ohio State University so that students could access and reference accurate textbook information, hence the name: PhaseWise.

## Features

- **PDF Processing**: Fetches and processes PDFs from a Google Drive folder.
- **Vector Search**: Uses FAISS (Facebook AI Similarity Search) to create a searchable vector database.
- **AI-Powered Responses**: Leverages OpenAI's GPT-3.5 (or any model of your choice) for contextual, chatbot-like answers.
- **Discord Integration**: Interact with the bot directly through Discord commands.

---

## Setup

### Prerequisites

1. **Google Cloud Service Account**:
   - Create and download a service account key file with access to Google Drive.
2. **OpenAI API Key**:
   - Sign up for OpenAI and generate an API key.
3. **Discord Bot Token**:
   - Create a bot on the [Discord Developer Portal](https://discord.com/developers/applications) and obtain the token.
4. **Environment Variables**:
   - Set up the following variables in your environment:
     - `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud credentials file.
     - `OPENAI_API_KEY`: OpenAI API key.
     - `DISCORD_TOKEN`: Discord bot token.
     - `DRIVE_FOLDER_ID`: ID of the Google Drive folder containing PDFs.

---

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phasewise.git
   cd phasewise

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Set up enviornment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
   export OPENAI_API_KEY=your_openai_api_key
   export DISCORD_TOKEN=your_discord_bot_token
   export DRIVE_FOLDER_ID=your_google_drive_folder_id

---

### Usage

1. Run the bot:
   ```bash
   python PhaseWiseBot.py

2. **Discord Commands:**
   !update: Update the FAISS vectorstore with the latest PDFs from Google Drive.

   !ask: Ask a question based on the processed knowledge base.
   
