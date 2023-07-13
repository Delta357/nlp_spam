import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Carregar os dados de treinamento
dados_treinamento = pd.read_csv('SPAM text message 20170820 - Data.csv')

# Pré-processamento dos dados de treinamento, se necessário
texto_treinamento = dados_treinamento['Message']

# Criar e ajustar o vetorizador aos dados de treinamento
vetorizador = CountVectorizer()
vetorizador.fit(texto_treinamento)

# Carregar o modelo a partir do arquivo .pkl
with open('modelo.pkl', 'rb') as file:
    modelo = pickle.load(file)
    
# Configurar a interface do usuário com Streamlit
st.title("Machine learning modelo Naive bayes")

st.header("Classificador SPAM")

# Texto
st.markdown("**Só texto em inglês o modelo só consegue interpretar idioma em inglês**")

# Carregar a imagem a partir de uma URL
image_url = 'https://img.freepik.com/free-vector/spam-isometric-landing-page-with-letter-envelopes-laptop-screen-electronic-email-service-messages-as-part-business-marketing-hacker-attack-webmail-malware-concept-3d-vector-web-banner_107791-10858.jpg?w=1380&t=st=1689130777~exp=1689131377~hmac=01044e343714eaa4d9a977784bbc573881f068a1bae942663461e2a557a43b89'
st.image(image_url, caption='Vetor')

# Função para realizar a classificação
def classificar(texto1, texto2, texto3):
    # Realizar pré-processamento dos textos, se necessário
    # Aplicar transformações necessárias nos textos, se necessário
    
    # Vetorizar os textos usando o vetorizador ajustado
    vetor_texto1 = vetorizador.transform([texto1])
    vetor_texto2 = vetorizador.transform([texto2])
    vetor_texto3 = vetorizador.transform([texto3])

    # Realizar a classificação usando o modelo carregado
    classe1 = modelo.predict(vetor_texto1)
    classe2 = modelo.predict(vetor_texto2)
    classe3 = modelo.predict(vetor_texto3)

    return classe1[0], classe2[0], classe3[0]

# Configurar o aplicativo Streamlit
def main():
    texto1 = st.text_area("Digite o texto do e-mail 1", height=100)
    texto2 = st.text_area("Digite o texto do e-mail 2", height=100)
    texto3 = st.text_area("Digite o texto do e-mail 3", height=100)

    if st.button("Classificar"):
        classe1, classe2, classe3 = classificar(texto1, texto2, texto3)
        st.write("Classificado é e-mail 1 é", classe1)
        st.write("Classificado é e-mail 2 é", classe2)
        st.write("Classificado é e-mail 3 é", classe3)

if __name__ == '__main__':
    main()
