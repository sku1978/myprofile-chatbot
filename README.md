# Shailesh Kumar Chatbot

This project builds a professional chatbot persona for Shailesh Kumar using OpenAI's GPT models. The bot answers questions based on Shailesh's CV and summary, provides evaluation feedback via a Gemini evaluator, and records user interactions through Pushover.

## Features

- ✅ Persona-based responses based on CV and professional summary
- ✅ Evaluation of responses using Gemini (Google's LLM API)
- ✅ Self-correction and re-generation upon rejection
- ✅ Recording of unknown questions and user contact details
- ✅ Gradio-based web interface

## Technologies Used

- Python 3.10+
- OpenAI API (chat completions)
- Google Gemini API (evaluation)
- Gradio (chat UI)
- Pushover (notification system)
- pypdf (for reading CV PDF)
- Pydantic (for schema validation)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
PUSHOVER_TOKEN=your_pushover_app_token
PUSHOVER_USER=your_pushover_user_key
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Folder Structure

```
.
├── me/
│   ├── ShaileshKumar_CV.pdf     # CV file used for context
│   └── summary.txt              # Short written summary of background
├── app.py                       # Main application logic
└── README.md                    # You're reading it!
```

## Running the App

```bash
python app.py
```

This will launch a Gradio web app where users can interact with the chatbot.

## License

MIT License. See [LICENSE](LICENSE) for details.