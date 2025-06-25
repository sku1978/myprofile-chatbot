# Import necessary libraries
from dotenv import load_dotenv                    # For loading environment variables from .env file
from openai import OpenAI                         # OpenAI SDK for accessing GPT models
import json                                       # For JSON parsing and serialization
import os                                         # For environment variable access
import requests                                   # For HTTP requests (used here for notifications)
from pypdf import PdfReader                       # To read text from PDF (used to extract CV)
import gradio as gr                               # Gradio for web UI
from pydantic import BaseModel                    # For typed data validation using Evaluation schema

# Load environment variables (such as API keys and tokens)
load_dotenv(override=True)

# Function to send a push notification using Pushover
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),   # App token
            "user": os.getenv("PUSHOVER_USER"),     # User key
            "message": text                         # Text message to send
        }
    )

# Function to record user contact details via a push notification
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

# Function to log unanswered or unrecognised user questions
def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

# JSON schema for the tool: record_user_details
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Additional info about the conversation"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

# JSON schema for the tool: record_unknown_question
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# List of tool definitions to pass to OpenAI
tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

# Define Evaluation model using Pydantic to validate LLM evaluation responses
class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


# Class representing the persona and behaviour of the AI agent (Shailesh Kumar)
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Shailesh Kumar"

        # Load CV text from PDF file
        reader = PdfReader("me/ShaileshKumar_CV.pdf")
        self.my_cv = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.my_cv += text

        # Load pre-written summary
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()
        
        # Gemini (Google) model for evaluating the agent's responses
        self.gemini = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    # Construct system prompt for the main agent (who speaks as Shailesh Kumar)
    def system_prompt(self):
        system_prompt = f"""You are acting as {self.name}. You are answering questions on {self.name}'s website,
particularly questions related to {self.name}'s career, background, skills and experience.
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible.
You are given a summary of {self.name}'s background CV profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer to any question, use your record_unknown_question tool.
If the user is engaging in discussion, steer them towards email contact and use the record_user_details tool.
Never share phone number or home address; use email only.
If the user has more than 5 questions, recommend continuing via email."""

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## MY Profile:\n{self.my_cv}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    # Construct system prompt for the evaluation agent (used to critique agent responses)
    def evaluator_system_prompt(self):
        evaluator_system_prompt = f"""You are an evaluator that decides whether a response to a question is acceptable.
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality.
The Agent is playing the role of {self.name} on their website and has been instructed to be professional and engaging.
Here is the information you have access to:"""
        evaluator_system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## CV Profile:\n{self.my_cv}\n\n"
        return evaluator_system_prompt

    # Create user prompt to provide context for evaluation
    def evaluator_user_prompt(self, reply, message, history):
        return (
            f"Here's the conversation between the User and the Agent:\n\n{history}\n\n"
            f"Latest message from the User:\n\n{message}\n\n"
            f"Agent's latest response:\n\n{reply}\n\n"
            f"Please evaluate the response for tone, accuracy, and usefulness."
        )

    # Evaluate agent's response using Gemini
    def evaluate(self, reply, message, history) -> Evaluation:
        messages = [
            {"role": "system", "content": self.evaluator_system_prompt()},
            {"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}
        ]
        response = self.gemini.beta.chat.completions.parse(
            model="gemini-2.0-flash", messages=messages, response_format=Evaluation
        )
        return response.choices[0].message.parsed

    # Re-run response generation after rejection with feedback
    def rerun(self, reply, message, history, feedback):
        updated_system_prompt = (
            self.system_prompt() +
            f"\n\n## Previous answer rejected\n"
            f"## Your attempted answer:\n{reply}\n\n"
            f"## Reason for rejection:\n{feedback}\n\n"
        )
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content

    # Handle tool calls from OpenAI response (e.g., record_unknown_question)
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    # Core function that processes user messages and generates chatbot responses
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False

        while not done:
            # Generate assistant response with potential tool use
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            reply = response.choices[0].message.content

            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls
                print(f"Tool calls detected: {tool_calls}", flush=True)

                tool_call_msg = {
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content,
                    "tool_calls": [tc.model_dump() for tc in tool_calls]
                }
                messages.append(tool_call_msg)

                results = self.handle_tool_call(tool_calls)
                messages.extend(results)
            else:
                evaluation = self.evaluate(reply, message, history)
                print(f"Evaluation: {evaluation.is_acceptable}, Feedback: {evaluation.feedback}", flush=True)
                if not evaluation.is_acceptable:
                    reply = self.rerun(reply, message, history, evaluation.feedback)
                done = True

        return response.choices[0].message.content


# Gradio web interface: Launches the chatbot UI
if __name__ == "__main__":
    me = Me()

    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown("### 🤖 You are talking to **Shailesh Kumar's** chatbot. You can ask me questions related to my professional career and expertise.")
        gr.ChatInterface(me.chat, type="messages")

    demo.launch()
