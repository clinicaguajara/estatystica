# 📦 IMPORTAÇÕES NECESSÁRIAS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import pathlib
import streamlit as st


# 💾 FUNÇÃO CACHEADA PARA LER O CONTEÚDO DE UM ARQUIVO CSS ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_css_file(css_path: str):
    """
    <docstrings> Lê o conteúdo de um arquivo CSS e retorna como string.

    Args:
        css_path (str): Caminho para o arquivo CSS a ser carregado.

    Calls:
        pathlib.Path():Construtor da classe Path para manipular caminhos no sistema de arquivos | instanciado por Path.
        pathlib.Path.exists(): Método do objeto Path para verificar a existência de um arquivo | instanciado por Path.
        open(): Função para abrir arquivos | built-in.
        logger.exception(): Método do objeto Logger para registra uma mensagem de erro junto com a stacktrace automática | instanciado por logger.

    Returns:
        str:
            Conteúdo do arquivo CSS como string.
            Retorna uma string vazia caso o arquivo não exista.
    """
    
    # Tenta ler o conteúdo do arquivo CSS...
    try:

        # Transforma uma variável do tipo string para um objeto do tipo pathlib.Path rico em métodos.
        path = pathlib.Path(css_path)
    
        # Se o caminho existir...
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="replace") as f:  # ⬅ Força leitura em UTF-8 em vez do cp1252 padrão do Windows.
                return f.read()                                             # ⬅ Retorna o conteúdo do arquivo CSS.
    
        # Caso contrário...
        else:
            return ""   # ⬅ Retorna uma string vazia como fallback.
        
    # Se houver exceções...
    except Exception as e:
        

        # Fallback de execução
        return ""


# 🎨 FUNÇÃO PARA INJETAR ESTILOS PERSONALIZADOS NA PÁGINA ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def load_css():
    """
    Injeta o conteúdo do arquivo CSS e estilos adicionais diretamente na página Streamlit.

    Args:
        None.

    Calls:
        load_css_file(): Função interna para ler e cachear o conteúdo de um arquivo CSS | definida em modules.design.py
        st.markdown(): Função para injetar código HTML (wrapper de método interno) | definida no módulo st.

    Returns:
        None:
            Apenas aplica o CSS na página atual.
    """
    
    # Obtém o conteúdo do arquivo CSS.
    css_content = load_css_file("assets/styles.css")

    # Se o conteúdo for encontrado...
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)   # ⬅ Aplica o CSS na página.
    
    # 🦄 Injeção de código HTML adicional (justifica o texto).
    st.markdown("""
        <style>
        summary {
            font-size: 24px !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)



