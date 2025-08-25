import re
import os
import argparse
from collections import defaultdict

def generate_genuine_pairs(dataset_dir, subject_regex, debug=False):
    """
    Gera um arquivo 'genuine_pair.txt' combinando arquivos de query e gallery
    com base em um ID de sujeito de 3 dígitos compartilhado em seus nomes.

    Argumentos:
        dataset_dir (str): O caminho para a raiz de uma pasta de dataset específica
                           (ex: 'TEST_DATA/SD258').
    """
    # 1. Define os caminhos para os diretórios de query e gallery
    # Usaremos os arquivos de minúcias, mas poderiam ser os de imagem também.
    query_dir = os.path.join(dataset_dir, 'mnt', 'query')
    gallery_dir = os.path.join(dataset_dir, 'mnt', 'gallery')

    # Verifica se os diretórios necessários existem
    if not os.path.isdir(query_dir) or not os.path.isdir(gallery_dir):
        print(f"ERRO: Não foi possível encontrar 'mnt/query' ou 'mnt/gallery' dentro de '{dataset_dir}'.")
        print("Por favor, certifique-se de que o dataset foi completamente organizado primeiro.")
        return

    # 2. Agrupa os nomes dos arquivos (sem extensão) pelo ID do sujeito extraído via regex
    print("Agrupando arquivos por ID de sujeito (via regex)...")
    query_map = defaultdict(list)
    gallery_map = defaultdict(list)

    subject_re = re.compile(subject_regex)

    # Processa os arquivos de query
    for filename in os.listdir(query_dir):
        match = subject_re.match(filename)
        if debug:
            print(f"Query: {filename} -> {match.groups() if match else 'NO MATCH'}")
        if match and len(match.groups()) == 1:
            subject_id = match.group(1)
            base_name, _ = os.path.splitext(filename)
            query_map[subject_id].append(base_name)
        else:
            print(f"Aviso: Pulando arquivo de query com formato inesperado: {filename}")

    # Processa os arquivos de galeria
    for filename in os.listdir(gallery_dir):
        match = subject_re.match(filename)
        if debug:
            print(f"Gallery: {filename} -> {match.groups() if match else 'NO MATCH'}")
        if match and len(match.groups()) == 1:
            subject_id = match.group(1)
            base_name, _ = os.path.splitext(filename)
            gallery_map[subject_id].append(base_name)
        else:
            print(f"Aviso: Pulando arquivo de galeria com formato inesperado: {filename}")

    # 3. Gera as combinações de pares genuínos
    print("Combinando pares com base nos IDs de sujeito...")
    genuine_pairs = []
    # Itera através dos IDs de sujeito encontrados nos arquivos de query
    for subject_id, query_files in query_map.items():
        # Verifica se o mesmo ID de sujeito também existe nos arquivos de galeria
        if subject_id in gallery_map:
            gallery_files = gallery_map[subject_id]
            # Cria todas as combinações possíveis para este sujeito
            for q_file in query_files:
                for g_file in gallery_files:
                    genuine_pairs.append(f"{q_file},{g_file}")

    if not genuine_pairs:
        print("Aviso: Nenhum par genuíno foi encontrado. Verifique os nomes dos arquivos e o conteúdo dos diretórios.")
        return

    # 4. Escreve o arquivo de saída
    output_filepath = os.path.join(dataset_dir, 'genuine_pairs.txt')
    print(f"Escrevendo {len(genuine_pairs)} pares em {output_filepath}...")

    try:
        with open(output_filepath, 'w') as f:
            # Ordena a lista para um resultado consistente
            genuine_pairs.sort()
            f.write('\n'.join(genuine_pairs))
        print("\n--- Geração do Arquivo de Pares Genuínos Concluída ---")
        print(f"Arquivo salvo com sucesso em: {output_filepath}")
    except Exception as e:
        print(f"ERRO ao escrever o arquivo de saída: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera um arquivo genuine_pair.txt para o projeto DMD.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help="Caminho para a raiz da pasta do dataset organizado.\nExemplo: TEST_DATA/SD258"
    )
    parser.add_argument(
        '--subject_regex',
        type=str,
        required=True,
        help="Regex para extrair o ID do sujeito dos nomes dos arquivos. Deve conter exatamente um grupo capturado. Exemplo: sd258_(\\d{3})_\\d+-\\d{2}_.*\\.mnt"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Ativa o modo de depuração."
    )

    args = parser.parse_args()
    generate_genuine_pairs(args.dataset_dir, args.subject_regex, args.debug if 'debug' in args else False)