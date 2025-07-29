# --- ESTÁGIO 1: O Construtor (Builder) ---
# Usamos a imagem completa para instalar tudo o que precisamos.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

# Instala o básico: Python e a ferramenta para criar ambientes virtuais (venv)
RUN apt-get update && \
    apt-get install -y python3.10-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Cria um ambiente virtual isolado. Esta é a chave!
# Todas as dependências e seus lixos ficarão contidos aqui.
RUN python3 -m venv /opt/venv

# Ativa o ambiente virtual para os próximos comandos
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

# Instala as dependências DENTRO do venv, sem guardar cache para economizar espaço
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- ESTÁGIO 2: A Imagem Final (Leve) ---
# Começamos DE NOVO com a mesma imagem base limpa.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# A imagem final precisa apenas do python3 para rodar
RUN apt-get update && \
    apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# A MÁGICA ACONTECE AQUI:
# Copiamos APENAS o ambiente virtual limpo do estágio "builder".
# Não trazemos o 'git', o 'pip', nem o cache de instalação.
COPY --from=builder /opt/venv /opt/venv

# Copiamos o nosso código-fonte
COPY src/ .

# Ativamos o ambiente virtual para o comando final
ENV PATH="/opt/venv/bin:$PATH"

# Comando para executar a aplicação usando o Python do nosso venv
CMD ["python", "main.py"]