from flask import *
from textsummarizer import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['input_text']
        
        if not request.form['nol']:
            nol = 3
        else:
            nol = int(request.form['nol'])
        
        try:
            summary, original_length = generate_summary(text, nol)
            return render_template('result.html', org_text=text, text_summary=summary, lines_original=original_length, lines_summary=nol)
        except Exception as e:
            custom_error_message = "Make sure that: The lines to summarize are not more than the number of lines in the text."
            return render_template('index.html', error_message=custom_error_message, input_text=text, nol=nol)

if __name__ == '__main__':
    app.run(debug=True)
