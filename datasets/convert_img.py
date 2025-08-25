import os
import shutil
import argparse
import re
from tqdm import tqdm

def parse_path(filepath, query_regex, query_value, debug = False):
    """
    Analisa o nome do arquivo usando regex e retorna 'query', 'gallery' ou None.
    Se o grupo capturado pelo regex for igual ao query_value, retorna 'query'.
    Caso contrário, retorna 'gallery'.
    Se não casar, retorna None.
    """
    filename = os.path.basename(filepath)
    match = re.match(query_regex, filename)
    if debug:
        print(filename, query_regex)
    if match:
        # O grupo capturado deve ser único
        if len(match.groups()) != 1:
            raise ValueError('O regex deve conter exatamente um grupo capturado.')
        group_val = match.group(1)
        print(group_val)
        if group_val == query_value:
            return 'query'
        else:
            return 'gallery'
    else:
        return None

def organize_fingerprint_dataset(source_dir, target_dir, dataset_name, query_regex, query_value, debug=False):
    """
    Organiza imagens de impressão digital de uma estrutura de origem específica
    para o formato exigido pelo projeto DMD.

    Argumentos:
        source_dir (str): O diretório raiz das imagens de origem.
                          (ex: '/localStorage/data/datasets/SD258/orig')
        target_dir (str): O diretório raiz para os dados organizados.
                          (ex: 'TEST_DATA')
        dataset_name (str): O nome para a nova pasta do dataset.
                            (ex: 'SD258')
    """
    # 1. Define os caminhos de destino com base na estrutura necessária
    query_path = os.path.join(target_dir, dataset_name, 'image', 'query')
    gallery_path = os.path.join(target_dir, dataset_name, 'image', 'gallery')

    # 2. Cria os diretórios de destino se eles não existirem
    print(f"Criando o diretório de destino para queries: {query_path}")
    os.makedirs(query_path, exist_ok=True)
    print(f"Criando o diretório de destino para galeria: {gallery_path}")
    os.makedirs(gallery_path, exist_ok=True)

    # 3. Encontra todos os arquivos .png para processar e criar uma barra de progresso
    files_to_process = []
    print("Procurando por arquivos de imagem...")
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print(f"Aviso: Nenhum arquivo .png foi encontrado em '{source_dir}'. Por favor, verifique o caminho.")
        return

    # 4. Processa e copia os arquivos
    print(f"\nForam encontradas {len(files_to_process)} imagens. Iniciando a cópia...")
    copied_queries = 0
    copied_galleries = 0

    for src_filepath in tqdm(files_to_process, desc="Organizando Imagens"):
        try:
            tipo = parse_path(src_filepath, query_regex, query_value, debug)
            filename = os.path.basename(src_filepath)
            if tipo == 'query':
                dest_filepath = os.path.join(query_path, filename)
                shutil.copy2(src_filepath, dest_filepath)
                copied_queries += 1
            elif tipo == 'gallery':
                dest_filepath = os.path.join(gallery_path, filename)
                shutil.copy2(src_filepath, dest_filepath)
                copied_galleries += 1
            # Outros valores de tipo serão ignorados
        except Exception as e:
            print(f"Erro ao processar o arquivo {src_filepath}: {e}")

    # 6. Exibe um resumo da operação
    print("\n--- Organização Concluída ---")
    print(f"Total de imagens de consulta (queries) copiadas: {copied_queries}")
    print(f"Total de imagens de galeria (gallery) copiadas: {copied_galleries}")
    print(f"Os dados para o dataset '{dataset_name}' estão prontos em: {os.path.join(target_dir, dataset_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organiza um dataset de impressões digitais para o formato do projeto DMD.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help="Diretório raiz das imagens de origem.\nExemplo: /localStorage/data/datasets/SD258/orig"
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='TEST_DATA',
        help="Diretório raiz onde os dados organizados serão salvos.\n(Padrão: TEST_DATA)"
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Nome para a pasta do novo dataset que será criada no diretório de destino.\nExemplo: SD258"
    )
    parser.add_argument(
        '--query_regex',
        type=str,
        required=True,
        help="Regex para identificar arquivos de query. Deve conter exatamente um grupo capturado, por exemplo: sd258_\\d+_\\d+-(\\d{2})_.*\\.png para capturar '00' ou '01' logo após o primeiro '-'."
    )
    parser.add_argument(
        '--query_value',
        type=str,
        required=True,
        help="Valor esperado no grupo capturado do regex para identificar queries. Exemplo: 00"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Ativa o modo de depuração."
    )

    args = parser.parse_args()

    organize_fingerprint_dataset(
        args.source_dir,
        args.target_dir,
        args.dataset_name,
        args.query_regex,
        args.query_value,
        args.debug if 'debug' in args else False
    )