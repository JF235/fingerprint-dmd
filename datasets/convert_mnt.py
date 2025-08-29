import re
import os
import argparse
from tqdm import tqdm

def parse_path(filename, query_regex, query_value, debug=False):
    """
    Analisa o nome do arquivo usando regex e retorna 'query', 'gallery' ou None.
    Se o grupo capturado pelo regex for igual ao query_value, retorna 'query'.
    Caso contrário, retorna 'gallery'.
    Se não casar, retorna None.
    """
    match = re.match(query_regex, filename)
    if debug:
        print(filename, query_regex, match)
    if match:
        if len(match.groups()) != 1:
            raise ValueError('O regex deve conter exatamente um grupo capturado.')
        group_val = match.group(1)
        if debug:
            print(group_val)
        if group_val == query_value:
            return 'query'
        else:
            return 'gallery'
    else:
        return None


def convert_minutiae_dummy_header(source_dir, target_dir, dataset_name, query_regex, query_value, debug=False):
    """
    Converte e organiza arquivos de minúcias de um formato TXT personalizado
    para o formato MNT, adicionando um cabeçalho com valores dummy (placeholders).

    Argumentos:
        source_dir (str): O diretório que contém os arquivos de minúcias .txt de origem.
        target_dir (str): O diretório raiz para os dados organizados (ex: 'TEST_DATA').
        dataset_name (str): O nome para a pasta do dataset (ex: 'SD258').
    """
    # 1. Define os caminhos de destino para os arquivos .mnt
    query_mnt_path = os.path.join(target_dir, dataset_name, 'mnt', 'query')
    gallery_mnt_path = os.path.join(target_dir, dataset_name, 'mnt', 'gallery')

    # 2. Cria os diretórios de destino
    print(f"Criando diretório para minúcias de query: {query_mnt_path}")
    os.makedirs(query_mnt_path, exist_ok=True)
    print(f"Criando diretório para minúcias de galeria: {gallery_mnt_path}")
    os.makedirs(gallery_mnt_path, exist_ok=True)

    # 3. Lista os arquivos de origem para processar
    try:
        source_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.txt')]
    except FileNotFoundError:
        print(f"ERRO: O diretório de origem '{source_dir}' não foi encontrado.")
        return

    if not source_files:
        print(f"Aviso: Nenhum arquivo .txt encontrado em '{source_dir}'. Verifique o caminho.")
        return

    # 4. Processa cada arquivo
    print(f"\nForam encontrados {len(source_files)} arquivos de minúcias para converter.")
    converted_queries = 0
    converted_galleries = 0
    
    # Valores dummy para o cabeçalho
    DUMMY_WIDTH = 500
    DUMMY_HEIGHT = 500


    for filename in tqdm(source_files, desc="Convertendo Minúcias"):
        source_filepath = os.path.join(source_dir, filename)
        base_name, _ = os.path.splitext(filename)
        new_filename = base_name + ".mnt"

        tipo = parse_path(filename, query_regex, query_value, debug)
        if tipo == 'query':
            target_filepath = os.path.join(query_mnt_path, new_filename)
            is_query = True
        elif tipo == 'gallery':
            target_filepath = os.path.join(gallery_mnt_path, new_filename)
            is_query = False
        else:
            continue

        try:
            with open(source_filepath, 'r') as infile, open(target_filepath, 'w') as outfile:
                # Escreve o cabeçalho DUMMY
                outfile.write(f"{DUMMY_WIDTH}\n")
                outfile.write(f"{DUMMY_HEIGHT}\n")
                outfile.write("0\n")
                outfile.write("0\n")
                outfile.write("\n")

                # Escreve os dados das minúcias
                minutiae_lines = []
                for line in infile:
                    if line.strip().startswith('#'):
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        x, y, theta = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        minutiae_lines.append(f"{x} {y} {theta}")
                
                outfile.write('\n'.join(minutiae_lines))

            if is_query:
                converted_queries += 1
            else:
                converted_galleries += 1
        except Exception as e:
            print(f"\nErro ao processar o arquivo {filename}: {e}")

    # 5. Exibe um resumo
    print("\n--- Conversão Concluída ---")
    print(f"Total de arquivos de query convertidos: {converted_queries}")
    print(f"Total de arquivos de galeria convertidos: {converted_galleries}")
    print(f"As minúcias para o dataset '{dataset_name}' estão prontas em: {os.path.join(target_dir, dataset_name, 'mnt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte arquivos de minúcias para o formato MNT com um cabeçalho DUMMY.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help="Diretório contendo os arquivos de minúcias .txt de origem."
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='TEST_DATA',
        help="Diretório raiz para os dados organizados.\n(Padrão: TEST_DATA)"
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Nome da pasta do dataset (deve ser o mesmo usado para as imagens).\nExemplo: SD258"
    )
    parser.add_argument(
        '--query_regex',
        type=str,
        required=True,
        help="Regex para identificar arquivos de query. Deve conter exatamente um grupo capturado, por exemplo: sd258_\\d+_\\d+-(\\d{2})_.*\\.txt para capturar '00' ou '01' logo após o primeiro '-'."
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
    convert_minutiae_dummy_header(
        args.source_dir,
        args.target_dir,
        args.dataset_name,
        args.query_regex,
        args.query_value,
        args.debug if 'debug' in args else False
    )