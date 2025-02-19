import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    nome,
    arquivo_log="app.log",
    nivel=logging.INFO,
    tamanho_max_mb=10,
    quantidade_backup=5,
):
    """
    Configura um objeto logger que escreve em arquivo e console.

    Args:
        nome (str): Nome do logger (normalmente __name__ do m dulo que chama esta fun o)
        arquivo_log (str): Caminho para o arquivo de log
        nivel (int): N vel de log (padr o: logging.INFO)
        tamanho_max_mb (int): Tamanho m ximo do arquivo de log em MB antes de rotacionar
        quantidade_backup (int): N mero de arquivos de backup a serem mantidos

    Retorna:
        logging.Logger: Inst ncia de logger configurada
    """
    # Cria o diretorio de logs se ele n o existe
    diretorio_logs = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(diretorio_logs, exist_ok=True)

    # Cria o caminho completo para o arquivo de log
    caminho_completo = os.path.join(diretorio_logs, arquivo_log)

    # Cria o logger
    logger = logging.getLogger(nome)
    logger.setLevel(nivel)

    # Cria formatadores
    formatador_arquivo = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Cria e configura o manipulador de arquivo com rota o
    manipulador_arquivo = RotatingFileHandler(
        caminho_completo,
        maxBytes=tamanho_max_mb * 1024 * 1024,  # Converte MB para bytes
        backupCount=quantidade_backup,
        encoding="utf-8",
    )
    manipulador_arquivo.setFormatter(formatador_arquivo)
    manipulador_arquivo.setLevel(nivel)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatador_arquivo)
    console_handler.setLevel(nivel)

    # Add handlers to logger if they haven't been added already
    if not logger.handlers:
        logger.addHandler(manipulador_arquivo)
        logger.addHandler(console_handler)

    return logger


# Example usage:
# logger = setup_logger(__name__, 'my_module.log')
# logger.info('This is an info message')
# logger.error('This is an error message')
