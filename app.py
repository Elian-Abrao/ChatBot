from flask import Flask, request, render_template, jsonify
from chatbot import generate_response, extract_text_from_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enviar_mensagem', methods=['POST'])
def enviar_mensagem():
    mensagem = request.form['mensagem']
    resposta_obj = generate_response(mensagem)
    resposta = extract_text_from_response(resposta_obj)
    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    app.run(debug=True)
