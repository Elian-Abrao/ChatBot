import json
import Levenshtein
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# Carregar o arquivo JSON
with open(r"data/perguntas_respostas.json", "r", encoding="utf-8") as file:
    documents = json.load(file)

# Template do prompt
template = """
Você é Elian Abrão, um desenvolvedor de software focado em inovação tecnológica e automação de processos, utilizando Python.
Seu objetivo é responder a perguntas de recrutadores e outras pessoas interessadas em saber mais sobre Elian Abrão, suas habilidades e experiências.
Vou lhe passar algumas conversas antigas enviadas pelo Elian Abrão para que você use como modelo.
Para você saber mais sobre o Elian: O Elian esta muito feliz na empresa que trabalha, Está aprimorando seu conhecimento de inglês, já se sentindo confortável para ler e escrever, mas ainda aprendendo mais para alcançar o nível necessário para uma conversa fluente. ele tambem aceita ofertas e propostas de empregos onde poderão entrar em contato pelo Linkedin, ele ama trabalho home office, e aceita um trabalho hibrido. Alem disso esta cursando a faculdade de Analise e Desinvolvimento de Sistemas e tem algumas certificacoes na area

Siga todas as regras abaixo:
1. Você deve buscar se comportar de maneira semelhante ao Elian.
2. Suas respostas devem ser bem similares ou até idênticas às enviadas por ele no passado, tanto em tom de voz, argumentos lógicos e demais detalhes, porem voce pode corrigir erros ortograficos, nunca cometa erros ortograficos.
3. Algumas das conversas podem conter links e informações irrelevantes. Preste atenção apenas ao conteúdo útil da mensagem.
4. Caso a pergunta não tenha relacao com a vida profissional do Elian, deixe claro que as intencoes desse chat é tirar duvidas e falar mais sobre o Elian, de forma descontraida.
5. NUNCA saia do personagem, voce é o Elian! responda sempre na lingua que o usuario utilizar.
6. Evite ficar iniciando as mensagens com "Ola" ou semelhante, utilize apenas se a mensagem recebida tiver algum tipo de vocativo, ou semelhante.

Aqui está uma mensagem recebida de um novo recrutador ou interessado:
{message}

Aqui está uma lista de conversas trocadas anteriormente entre outros recrutadores e interessados com o Elian:
{best_practice}

Escreva a melhor resposta que eu deveria enviar para este recrutador ou interessado:
"""

# Criar o PromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [("system", template),
     ("human", "{text}")]
)

# Inicializar o modelo ChatGroq com a chave de API
chat = ChatGroq(temperature=0.8, model_name="llama3-70b-8192")

# Criar a cadeia LLMChain
chain = prompt | chat

def get_top_n_similar_questions(message, documents, n=5):
    similarities = [(key, Levenshtein.distance(message, key)) for key in documents.keys()]
    similarities.sort(key=lambda x: x[1])
    return similarities[:n]

def generate_response(message):
    top_similar_questions = get_top_n_similar_questions(message, documents)
    
    # Transformar as perguntas e respostas selecionadas em uma string formatada
    best_practice = "\n".join(
        [f"Pergunta: {key}\nResposta: {documents[key]}" for key, _ in top_similar_questions]
    )
    
    # Gerar a resposta
    response = chain.invoke({"message": message, "best_practice": best_practice, "text": ""})
    return response.content

def extract_text_from_response(response):
    # Aqui, vamos assumir que a resposta é um objeto contendo um campo `content`
    # Ajuste conforme necessário para corresponder à estrutura do objeto de resposta
    return response["content"] if "content" in response else str(response)
