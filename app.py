from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return open("docs/index.html").read()

if __name__ == '__main__':
    app.run(debug=True, port=8080)
