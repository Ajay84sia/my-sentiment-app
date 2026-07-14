import gradio as gr
from transformers import pipeline

# Load the AI model
model = pipeline('sentiment-analysis')

# This function runs when the user clicks Submit
def analyze_sentiment(text):
    result = model(text)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)

    if label == 'POSITIVE':
        return f'😊 POSITIVE — The AI is {score}% confident!'
    else:
        return f'😞 NEGATIVE — The AI is {score}% confident!'

# Build the app interface
app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Enter Text",
        placeholder="Type a sentence here..."
    ),
    outputs=gr.Textbox(label='Your Result'),
    title='😊 Sentiment Analyzer App 😊',
    description='Type any sentence and I will tell you if it is Positive or Negative!',
    article="""
    <div style="text-align:center; margin-top:20px; font-size:16px;">
        Designed and built with ❤️ by
        <a href="https://www.linkedin.com/in/ajay-84sia/"
           target="_blank"
           style="text-decoration:none; font-weight:bold;">
           Ajay Chaurasia (Ajay84sia)
        </a>
    </div>
    """
)

# Launch the app
import os

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
