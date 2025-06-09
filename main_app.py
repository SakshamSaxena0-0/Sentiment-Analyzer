from transformers import pipeline
import gradio as gr

sentiment = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """
    Takes input text and returns the
    predicted label and confidence.
    """
    result = sentiment(text)[0]
    label = result["label"]
    score = result["score"]
    return {"label": label, "confidence": f"{score:.2f}"}

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter some text..."),
    outputs=[
        gr.Label(num_top_classes=2, label="Sentiment"),
        gr.Textbox(label="Confidence")
    ],
    title="🤗 Sentiment Analyzer",
    description="Enter text and get back its sentiment (POSITIVE/NEGATIVE) with confidence score."
)

if __name__ == "__main__":
    iface.launch()
