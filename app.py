from flask import Flask, render_template, request
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
from emoji_map import emoji_map


# Initialize Flask app
app = Flask(__name__)

# Load emotion classifier
print("Loading model...")
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
print("Model loaded.")


def replace_emojis(text):
    return ''.join([emoji_map.get(ch, ch) for ch in text])


# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    chart_path = None

    if request.method == 'POST':
        try:
            text = request.form.get('text', '').strip()
            if not text:
                return render_template('index.html', sentiment="Please enter some text.")

            # Replace emojis with words
            text = replace_emojis(text)

            # Emotion classification
            results = classifier(text)[0]
            results.sort(key=lambda x: x['score'], reverse=True)
            top_emotion = results[0]['label']
            emotion = f"Detected Emotion: {top_emotion.capitalize()}"

            # Plot results
            labels = [res['label'].capitalize() for res in results]
            scores = [res['score'] for res in results]

            plt.figure(figsize=(6, 4))
            plt.bar(labels, scores, color='skyblue')
            plt.title('Emotion Scores')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()

            chart_path = os.path.join('static', 'sentiment.png')
            plt.savefig(chart_path)
            plt.close()

        except Exception as e:
            print("Error:", e)
            return "Internal Server Error", 500

    return render_template('index.html', sentiment=emotion, chart=chart_path)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
