import re
import numpy as np
from fractions import Fraction
from dataclasses import dataclass

@dataclass
class ProblemaPL:
    quantidade_variaveis: int
    vetor_variaveis: np.ndarray
    vetor_coeficientes: np.ndarray
    matriz_coeficientes: np.ndarray
    vetor_b: np.ndarray
    vetor_operadores: np.ndarray
    vetor_operadores_novo: np.ndarray

# Interpretação do arquivo.txt ########################################################################################################################################################################################################
def contar_variaveis_e_vetor(arquivo):
    variaveis = set()  # Usar um set para evitar duplicação
    with open(arquivo, 'r', encoding="utf-8") as f:
        for linha in f:
            # Remove as palavras "max", "s.a." e "livre" da linha
            linha_sem_palavras = re.sub(r'\b(max|s\.a\.|livre|min|s\.a\. )\b', '', linha)
            
            # Regex para extrair as variáveis, assumindo que elas são compostas apenas por letras
            termos = re.findall(r'[a-zA-Z]+', linha_sem_palavras)
            
            # Adiciona as variáveis ao set
            variaveis.update(termos)
    
    # Converte o set para uma lista (vetor de variáveis) e ordena
    vetor_variaveis = np.array(sorted(list(variaveis)))
    
    # Retorna a quantidade de variáveis e o vetor de variáveis
    return len(variaveis), vetor_variaveis

def obter_vetor_coeficientes(arquivo, vetor_variaveis):
    with open(arquivo, 'r', encoding="utf-8") as f:
        primeira_linha = f.readline().strip()  # Lê a primeira linha

    # Verifica se a função objetivo é "max" ou "min"
    match = re.match(r'^(max|min)\s+', primeira_linha)
    if match:
        tipo_objetivo = match.group(1)  # Armazena se era "max" ou "min"
        primeira_linha = re.sub(r'^(max|min)\s+', '', primeira_linha)  # Remove "max" ou "min"

    if tipo_objetivo == "max":
        tipo = 1
    else:
        tipo = 0

    # Inicializa o vetor de coeficientes como zeros
    coeficientes = np.zeros(len(vetor_variaveis), dtype=float)

    # Encontra os coeficientes e as variáveis
    termos = re.findall(r'([+-]?\s*\d*\.*\d*)\s*([a-zA-Z]+)', primeira_linha)

    for coef, var in termos:
        coef = coef.replace(" ", "")  # Remove espaços em branco
        if coef in ["", "+"]:  # Caso o coeficiente esteja ausente, assume 1
            coef = "1"
        elif coef == "-":  # Caso seja apenas "-", assume -1
            coef = "-1"
        
        coef = float(coef)  # Converte para número
        
        # Atualiza a posição correta no vetor de coeficientes
        if var in vetor_variaveis:
            index = np.where(vetor_variaveis == var)[0][0]
            coeficientes[index] = coef

    return coeficientes, tipo

def obter_matriz_coeficientes_restricoes(arquivo, vetor_variaveis):
    # Lê as linhas do arquivo
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    # Inicializa a matriz de coeficientes das restrições
    matriz_coeficientes = []

    # A partir da segunda linha, processa as restrições
    for linha in linhas[1:]:  # Ignora a primeira linha (função objetivo)
        # Remove quebras de linha e espaços extras
        linha = linha.strip()

        # Remove o prefixo "s.a." (ou qualquer outra coisa que anteceda a restrição)
        linha = re.sub(r'^\s*(s\.a\.|livre|subject\s+to)\s*', '', linha, flags=re.IGNORECASE)

        # Ignora restrições que são do tipo "variável livre", como "x <= 0", "y >= 0"
        if re.search(r'^[a-zA-Z]+\s*(<=|>=|=|>|<)\s*0', linha):  
            continue  # Ignora restrições de variáveis livres

        # Identifica o tipo de restrição (<=, >=, =)
        if '<=' in linha or '<' in linha:
            tipo_restricao = '<=|<'
        elif '>=' in linha or '>' in linha:
            tipo_restricao = '>=|>'
        elif '=' in linha:
            tipo_restricao = '='
        else:
            continue  # Linha inválida (sem restrição)

        # Divide a linha em termos
        termos = re.findall(r'([+-]?\s*\d*\.*\d*)\s*([a-zA-Z]+)', linha)

        # Inicializa um vetor de coeficientes para esta restrição
        coeficientes_restricao = np.zeros(len(vetor_variaveis), dtype=float)

        # Preenche o vetor de coeficientes
        for coef, var in termos:
            coef = coef.strip()
            coef = coef.replace(" ", "")

            # Caso o coeficiente seja vazio ou apenas o sinal
            if coef in ["", ".", "+"]:
                coef = "1"  # Define 1, pois 'x' é implicitamente 1x
            elif coef == "-":
                coef = "-1"  # Define -1, pois '-x' é implicitamente -1x
            elif coef == ".":
                coef = "0"  # Caso seja apenas um ponto, substituímos por 0
            else:
                coef = coef.replace(" ", "")  # Remove espaços extras
                try:
                    
                    coef = float(coef)  # Tenta converter o coeficiente para número
                except ValueError:
                    coef = 0.0  # Se não for um número válido, define como 0

            # Encontra o índice da variável no vetor de variáveis
            indices = np.where(vetor_variaveis == var)[0]
            if len(indices) > 0:
                index = indices[0]  # Assume que a variável está no vetor
                coeficientes_restricao[index] = coef

        # Adiciona o vetor de coeficientes dessa restrição à matriz
        matriz_coeficientes.append(coeficientes_restricao)

    # Converte a lista de listas para uma matriz NumPy
    return np.array(matriz_coeficientes)

def obter_vetor_b(arquivo):
    # Lê as linhas do arquivo
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    # Inicializa o vetor b
    vetor_b = []

    # A partir da segunda linha, processa as restrições
    for linha in linhas[1:]:  # Ignora a primeira linha (função objetivo)
        linha = linha.strip()

        # Ignora prefixos como "s.a.", "livre", etc., mas não interfere nas restrições
        linha = re.sub(r'^\s*(s\.a\.|livre|subject\s+to)\s*', '', linha, flags=re.IGNORECASE)

        # Ignora restrições de variáveis livres como "x <= 0", "y >= 0" (apenas essas)
        if re.search(r'^[a-zA-Z]+\s*(<=|>=|=|<|>)\s*0$', linha):  
            continue  # Ignora essas restrições específicas

        # Verifica se há um número no lado direito da restrição e extrai (suportando frações e decimais)
        match = re.search(r'(<=|>=|=|<|>)\s*(-?\d+(?:\.\d+)?(?:/\d+)?|\d+/\d+)', linha)
        if match:
            valor_str = match.group(2).replace(" ", "")  # Remove espaços

            # Converte frações corretamente
            if "/" in valor_str:
                valor_b = float(Fraction(valor_str))  # Converte fração para float
            else:
                valor_b = float(valor_str)  # Converte número decimal normalmente
            
            vetor_b.append(valor_b)
    # Retorna o vetor de b como um array NumPy
    return np.array(vetor_b)

def obter_operadores_restricoes(arquivo):
    # Lê as linhas do arquivo
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    # Inicializa o vetor de operadores
    operadores = []

    # A partir da segunda linha, processa as restrições
    for linha in linhas[1:]:  # Ignora a primeira linha (função objetivo)
        linha = linha.strip()

        # Ignora prefixos como "s.a.", "livre", etc., mas não interfere nas restrições
        linha = re.sub(r'^\s*(s\.a\.|livre|subject\s+to)\s*', '', linha, flags=re.IGNORECASE)

        # Ignora restrições de variáveis livres como "x <= 0", "y >= 0" (apenas essas)
        if re.search(r'^[a-zA-Z]+\s*(<=|>=|=|<|>)\s*0$', linha):  
            continue  # Ignora essas restrições específicas

        # Verifica se há operadores de comparação
        match = re.search(r'(<=|>=|=|<|>)', linha)
        if match:
            operador = match.group(1)  # Extrai o operador encontrado
            operadores.append(operador)

    # Retorna o vetor de operadores como um array NumPy
    return np.array(operadores)

def gerar_formato_matricial(arquivo):
    quantidade_variaveis, vetor_variaveis = contar_variaveis_e_vetor(arquivo)
    vetor_coeficientes, tipo = obter_vetor_coeficientes(arquivo, vetor_variaveis)
    matriz_coeficientes = obter_matriz_coeficientes_restricoes(arquivo, vetor_variaveis)
    vetor_b = obter_vetor_b(arquivo)
    vetor_operadores = obter_operadores_restricoes(arquivo)

    var_livres = verificar_variaveis_livres(arquivo, vetor_variaveis)
    var_sinais = verificar_variaveis_sinal(arquivo, vetor_variaveis)
    print("\n Problema dado:\n")
    resultado = gerar_formato_textual(matriz_coeficientes, vetor_operadores, vetor_b, vetor_coeficientes, vetor_variaveis, var_livres, var_sinais, tipo)
    print(resultado)

    ProblemaPL.quantidade_variaveis = quantidade_variaveis
    ProblemaPL.vetor_variaveis = vetor_variaveis
    ProblemaPL.vetor_coeficientes = vetor_coeficientes
    ProblemaPL.matriz_coeficientes = matriz_coeficientes
    ProblemaPL.vetor_b = vetor_b
    ProblemaPL.vetor_operadores = vetor_operadores

    return ProblemaPL
# Transformar para forma padrão ##################################################################################################################################################################################################
def transformar_para_min(arquivo, vetor_coeficientes):
    with open(arquivo, 'r', encoding="utf-8") as f:
        primeira_linha = f.readline().strip()  # Lê a primeira linha

    # Verifica se a função objetivo é "max" ou "min"
    match = re.match(r'^(max|min)\s+', primeira_linha)
    if match:
        tipo_objetivo = match.group(1)  # Armazena se era "max" ou "min"
        primeira_linha = re.sub(r'^(max|min)\s+', '', primeira_linha)  # Remove "max" ou "min"

    # Se o problema era de maximização, multiplica os coeficientes por -1
    if tipo_objetivo == "max":
        vetor_coeficientes[vetor_coeficientes != 0] *= -1  # Apenas elementos diferentes de zero são multiplicados

    return vetor_coeficientes

def tratar_variaveis_de_sinal(arquivo, matriz_coeficientes, vetor_variaveis, vetor_coef_objetivo):
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    # Converte vetor de variáveis para lista para permitir modificações
    novas_variaveis = list(vetor_variaveis)
    matriz_modificada = matriz_coeficientes.copy()
    vetor_coef_objetivo_modificado = vetor_coef_objetivo.copy()
    
    # Percorre as linhas para encontrar restrições de sinal no formato "x <= 0"
    for linha in linhas:
        linha = linha.strip()
        
        # Identifica variáveis com restrições do tipo "x <= 0"
        match = re.match(r'^([a-zA-Z]+)\s*(<=|<)\s*0$', linha)
        if match:
            variavel = match.group(1)  # Captura a variável

            # Verifica se a variável está no vetor de variáveis
            if variavel in novas_variaveis:
                indice = novas_variaveis.index(variavel)  # Pega o índice na lista

                # Substitui a variável por uma nova variável x'
                novas_variaveis[indice] = f"{variavel}'"
                
                # Multiplica a coluna correspondente na matriz de coeficientes por -1, se houver valores diferentes de zero
                if np.any(matriz_modificada[:, indice] != 0):
                    matriz_modificada[:, indice] *= -1

                # Multiplica o coeficiente correspondente na função objetivo por -1, se for diferente de zero
                if vetor_coef_objetivo_modificado[indice] != 0:
                    vetor_coef_objetivo_modificado[indice] *= -1

    # Substituir -0.0 por 0.0 para manter a clareza
    matriz_modificada[matriz_modificada == -0.0] = 0.0
    vetor_coef_objetivo_modificado[vetor_coef_objetivo_modificado == -0.0] = 0.0
    # Retorna as variáveis como array NumPy novamente, junto com as variáveis de sinal alterado
    return matriz_modificada, np.array(novas_variaveis), vetor_coef_objetivo_modificado

def tratar_variaveis_livres(arquivo, matriz_coeficientes, vetor_variaveis, vetor_coef_objetivo):
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    novas_variaveis = list(vetor_variaveis)
    matriz_modificada = matriz_coeficientes.copy()
    vetor_coef_objetivo_modificado = vetor_coef_objetivo.copy()

    # Percorre as linhas para encontrar variáveis livres no formato "z livre"
    for linha in linhas:
        linha = linha.strip()
        
        match = re.match(r'^([a-zA-Z]+)\s+livre$', linha, re.IGNORECASE)
        if match:
            variavel = match.group(1)  # Captura a variável

            # Verifica se a variável está no vetor de variáveis
            if variavel in novas_variaveis:
                indice = novas_variaveis.index(variavel)  # Pega o índice na lista

                # Define novas variáveis z* e z**
                var_estrela = f"{variavel}*"
                var_duplo_estrela = f"{variavel}**"

                # Substitui a variável original pelas duas novas variáveis
                novas_variaveis[indice] = var_estrela  # Substitui a variável antiga por z*
                novas_variaveis.append(var_duplo_estrela)  # Adiciona z**

                # Adiciona uma nova coluna para z**
                nova_coluna = np.zeros((matriz_modificada.shape[0], 1))  # Coluna zerada inicialmente
                matriz_modificada = np.hstack((matriz_modificada, nova_coluna))  # Adiciona nova coluna

                # Modifica a matriz de coeficientes para z = z* - z**
                matriz_modificada[:, indice] *= 1  # Mantém coeficientes de z* como estão
                matriz_modificada[:, -1] = -matriz_modificada[:, indice]  # Define coeficientes de z** como -z*

                # Adiciona um coeficiente 0 para z** na função objetivo
                vetor_coef_objetivo_modificado = np.append(vetor_coef_objetivo_modificado, -vetor_coef_objetivo_modificado[indice])

    # Substituir -0.0 por 0.0 para manter a clareza
    matriz_modificada[matriz_modificada == -0.0] = 0.0
    vetor_coef_objetivo_modificado[vetor_coef_objetivo_modificado == -0.0] = 0.0
    
    return matriz_modificada, np.array(novas_variaveis), vetor_coef_objetivo_modificado

def verificar_variaveis_livres(arquivo, vetor_variaveis):
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    # Lista para armazenar as variáveis livres
    variaveis_livres = []

    # Percorre as linhas para encontrar variáveis livres no formato "z livre"
    for linha in linhas:
        linha = linha.strip()
        
        # Verifica se a linha corresponde ao formato "variavel livre"
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s+livre$', linha, re.IGNORECASE)
        if match:
            variavel = match.group(1)  # Captura a variável
            if variavel in vetor_variaveis:
                variaveis_livres.append(variavel)
    
    return np.array(variaveis_livres)

def verificar_variaveis_sinal(arquivo, vetor_variaveis):
    with open(arquivo, 'r', encoding="utf-8") as f:
        linhas = f.readlines()

    # Lista para armazenar as variáveis com restrições de sinal
    variaveis_de_sinal = []

    # Percorre as linhas para encontrar restrições no formato "x <= 0" ou "x < 0"
    for linha in linhas:
        linha = linha.strip()
        
        # Verifica se a linha corresponde ao formato "variavel <= 0" ou "variavel < 0"
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|<)\s*0$', linha)
        if match:
            variavel = match.group(1)  # Captura a variável
            if variavel in vetor_variaveis:
                variaveis_de_sinal.append(variavel)
    
    return np.array(variaveis_de_sinal)

def ajustar_vetor_b(matriz_coeficientes, vetor_b, operadores):
    matriz_ajustada = matriz_coeficientes.copy()
    vetor_b_ajustado = vetor_b.copy()
    operadores_ajustados = operadores.copy()  # Copia o vetor de operadores

    for i in range(len(vetor_b_ajustado)):
        if vetor_b_ajustado[i] < 0:
            # Multiplica a linha correspondente na matriz por -1
            matriz_ajustada[i, :] *= -1
            # Multiplica o valor de b por -1
            vetor_b_ajustado[i] *= -1
            # Troca o operador '>=', '<='
            if operadores_ajustados[i] == '>=':
                operadores_ajustados[i] = '<='
            elif operadores_ajustados[i] == '<=':
                operadores_ajustados[i] = '>='
            elif operadores_ajustados[i] == '<':
                operadores_ajustados[i] = '>'
            elif operadores_ajustados[i] == '>':
                operadores_ajustados[i] = '<'

    # Substituir -0.0 por 0.0 para manter a clareza
    matriz_ajustada[matriz_ajustada == -0.0] = 0.0
    vetor_b_ajustado[vetor_b_ajustado == -0.0] = 0.0

    return matriz_ajustada, vetor_b_ajustado, operadores_ajustados

def adicionar_variavel_folga(matriz_coeficientes, vetor_variaveis, restricoes, coef_objetivo):
    # Converte para lista para adicionar variáveis
    novas_variaveis = vetor_variaveis.tolist()
    num_linhas = matriz_coeficientes.shape[0]

    # Inicializa matriz identidade para as variáveis de folga
    identidade = np.zeros((num_linhas, 0))  # Começa sem colunas

    operadores_ajustados = restricoes.copy()  # Copia o vetor de operadores

    count = 0
    for i, restricao in enumerate(restricoes):
        if restricao == '<=' or restricao == '<':
            count += 1
            nova_coluna = np.zeros((num_linhas, 1))  # Cria nova coluna
            nova_coluna[i, 0] = 1  # Define o 1 na posição correta
            identidade = np.hstack((identidade, nova_coluna))  # Adiciona à matriz identidade
            novas_variaveis.append(f"f_{count}")  # Adiciona a variável de folga

            # Adiciona 0 no vetor de coeficiente da função objetiva para a variável de folga
            coef_objetivo = np.append(coef_objetivo, 0)
            operadores_ajustados[i] = "="

    # Se pelo menos uma coluna foi adicionada, concatena com a matriz de coeficientes
    if identidade.shape[1] > 0:
        matriz_coeficientes = np.hstack((matriz_coeficientes, identidade))

    return matriz_coeficientes, np.array(novas_variaveis), coef_objetivo, operadores_ajustados

def adicionar_variavel_excesso(matriz_coeficientes, vetor_variaveis, restricoes, coef_objetivo):
    # Converte para lista para adicionar variáveis
    novas_variaveis = vetor_variaveis.tolist()
    num_linhas = matriz_coeficientes.shape[0]

    # Inicializa matriz identidade para as variáveis de excesso
    identidade = np.zeros((num_linhas, 0))  # Começa sem colunas

    operadores_ajustados = restricoes.copy()  # Copia o vetor de operadores

    count = 0
    for i, restricao in enumerate(restricoes):
        if restricao == '>=' or restricao == '>':
            count += 1
            nova_coluna = np.zeros((num_linhas, 1))  # Cria nova coluna
            nova_coluna[i, 0] = -1  # Define o 1 na posição correta
            identidade = np.hstack((identidade, nova_coluna))  # Adiciona à matriz identidade
            novas_variaveis.append(f"e_{count}")  # Adiciona a variável de excesso

            # Adiciona 0 no vetor de coeficiente da função objetiva para a variável de excesso
            coef_objetivo = np.append(coef_objetivo, 0)
            operadores_ajustados[i] = "="

    # Se pelo menos uma coluna foi adicionada, concatena com a matriz de coeficientes
    if identidade.shape[1] > 0:
        matriz_coeficientes = np.hstack((matriz_coeficientes, identidade))

    return matriz_coeficientes, np.array(novas_variaveis), coef_objetivo, operadores_ajustados

# Gerar PL extra #######################################################################################################################################################################################################
def possui_solucao_basica_viavel(matriz_coeficientes):
    num_linhas, num_colunas = matriz_coeficientes.shape
    identidade = np.eye(num_linhas)  # Matriz identidade do tamanho adequado

    colunas_base = []  # Lista para armazenar os índices das colunas que formam a identidade
    colunas_nao_encontradas = []

    for i in range(num_linhas):  # Percorre as linhas da identidade (primeira coluna, depois a segunda, etc.)
        coluna_encontrada = False
        for j in range(num_colunas):  # Percorre as colunas da matriz de coeficientes
            if j not in colunas_base:  # Garante que não repete colunas já usadas
                if np.allclose(matriz_coeficientes[:, j], identidade[:, i]):  # Compara a coluna com a correspondente na identidade
                    colunas_base.append(j)  # Salva o índice da coluna encontrada
                    coluna_encontrada = True
                    break  # Passa para a próxima linha da identidade
        if not coluna_encontrada:  # Se não encontrou uma correspondência para a coluna da identidade
            colunas_nao_encontradas.append(i)  # Salva o índice da coluna da identidade que não foi encontrada
    # Se encontrou todas as colunas na ordem correta, retorna sucesso
    if len(colunas_base) == num_linhas:
        return True, colunas_base, colunas_nao_encontradas
    
    return False, [], colunas_nao_encontradas  # Retorna falso se não encontrou todas as colunas necessárias

def variaveis_basicas(indices_colunas, vetor_variaveis):
    vetor_variaveis = vetor_variaveis.tolist()  # Converte para lista, se for um array NumPy
    return [vetor_variaveis[idx] for idx in indices_colunas]

def adicionar_variavel_artificial(matriz_coeficientes, vetor_variaveis, restricoes, resultado, coef_objetivo, colunas_nao_encontradas):
    # Converte para lista para adicionar variáveis
    novas_variaveis = vetor_variaveis.tolist()
    num_linhas = matriz_coeficientes.shape[0]

    # Inicializa matriz identidade para as variáveis de folga
    identidade = np.zeros((num_linhas, 0))  # Começa sem colunas
    variaveis_artificiais = []
    count = 0
    for i, restricao in enumerate(restricoes):
        if not resultado:
            if restricao == '>=' or restricao == '>' or restricao == '=':
                if i in colunas_nao_encontradas:
                    count += 1
                    nova_coluna = np.zeros((num_linhas, 1))  # Cria nova coluna
                    nova_coluna[i, 0] = 1  # Define o 1 na posição correta
                    identidade = np.hstack((identidade, nova_coluna))  # Adiciona à matriz identidade
                    # O índice da variável artificial será após as variáveis originais e de folga
                    indice_var_artificial = len(novas_variaveis)
                    variaveis_artificiais.append(indice_var_artificial)
                    
                    novas_variaveis.append(f"a_{count}")  # Adiciona a variável artificial

                    # Adiciona 0 no vetor de coeficiente da função objetiva para a variável de excesso
                    coef_objetivo = np.append(coef_objetivo, 0)

    # Se pelo menos uma coluna foi adicionada, concatena com a matriz de coeficientes
    if identidade.shape[1] > 0:
        matriz_coeficientes = np.hstack((matriz_coeficientes, identidade))

    return matriz_coeficientes, np.array(novas_variaveis), coef_objetivo, variaveis_artificiais

def gerar_formato_textual(matriz_coeficientes, vetor_operadores, vetor_b, coeficientes_objetivo, vetor_variaveis, variaveis_livres, variaveis_sinais, tipo):
    # Função objetivo
    objetivo = "max" if tipo else "min"
    objetivo_str = f"{objetivo} " + " + ".join(
        [f"{int(coef) if coef.is_integer() else coef}{var}" if coef >= 0 
        else f"- {-int(coef) if coef.is_integer() else -coef}{var}" 
        for coef, var in zip(coeficientes_objetivo, vetor_variaveis)]
    )

    # Restrições
    restricoes_str = "s.a.\n"
    for i in range(matriz_coeficientes.shape[0]):
        restricao = " ".join(
            [f"+ {int(coef) if coef.is_integer() else coef}{var}" if coef >= 0 
            else f"- {-int(coef) if coef.is_integer() else -coef}{var}"
            for coef, var in zip(matriz_coeficientes[i], vetor_variaveis)]
        )
        operador = vetor_operadores[i]
        b_valor = vetor_b[i]
        restricoes_str += f"    {restricao} {operador} {b_valor}\n"

    variaveis_str = ""
    for var in vetor_variaveis:
        # Verifica se a variável é livre
        if var in variaveis_livres:
            variaveis_str += f"{var} livre\n"
        # Verifica se a variável é de sinal (não negativa)
        elif var in variaveis_sinais:
            variaveis_str += f"{var} <= 0\n"
        else:
            variaveis_str += f"{var} >= 0\n"  # Caso padrão, pode ser ajustado conforme necessário

    objetivo_str = objetivo_str.replace("+ -", "-")
    restricoes_str = restricoes_str.replace("+ -", "-")
    variaveis_str = variaveis_str.replace("+ -", "-")

    # Substitui 0 seguido de qualquer variável (ex: 0f_5) por espaços
    objetivo_str = re.sub(r"([+-]?\s*0)([a-zA-Z_][a-zA-Z0-9_*'\s]*)", 
                     lambda match: ('  ' if '+' not in match.group(0) else '') + ' ' * len(match.group(0)), 
                     objetivo_str)
    restricoes_str = re.sub(r"([+-]?\s*0)([a-zA-Z_][a-zA-Z0-9_*'\s]*)", 
                     lambda match: ('\n ' if '+' not in match.group(0) else '') + ' ' * len(match.group(0)), 
                     restricoes_str)
    variaveis_str = re.sub(r"\+?\s*0([a-zA-Z_][a-zA-Z0-9_*'\s]*)", lambda match: ' ' * (len(match.group(0))), variaveis_str)

    return f"{objetivo_str}\n{restricoes_str}{variaveis_str}"

def vetor_coef_pl_auxiliar(vetor_variaveis):
    coeficientes_objetivo = np.zeros(len(vetor_variaveis))

    for i, var in enumerate(vetor_variaveis):
        if var.startswith("a_"):  # Variáveis de folga e artificiais
            coeficientes_objetivo[i] = 1

    return coeficientes_objetivo

########################################################################################################################################################################################################
def transformar_para_forma_padrao(problema: ProblemaPL, arquivo: str) -> ProblemaPL:
    vetor_coef_tratado = transformar_para_min(arquivo, problema.vetor_coeficientes)
    matriz_tratada, novas_variaveis, vetor_coef_tratado = tratar_variaveis_de_sinal(
        arquivo, problema.matriz_coeficientes, problema.vetor_variaveis, vetor_coef_tratado
    )
    matriz_tratada, novas_variaveis, vetor_coef_tratado = tratar_variaveis_livres(
        arquivo, matriz_tratada, novas_variaveis, vetor_coef_tratado
    )
    matriz_tratada, novo_vetor_b, novo_vetor_operadores = ajustar_vetor_b(
        matriz_tratada, problema.vetor_b, problema.vetor_operadores
    )
    matriz_tratada, novas_variaveis, vetor_coef_tratado, novo_vetor_operadores = adicionar_variavel_folga(
        matriz_tratada, novas_variaveis, novo_vetor_operadores, vetor_coef_tratado
    )
    matriz_tratada, novas_variaveis, vetor_coef_tratado, novo_vetor_operadores = adicionar_variavel_excesso(
        matriz_tratada, novas_variaveis, novo_vetor_operadores, vetor_coef_tratado
    )

    print("\nTransformando para forma padrão:\n")
    resultado = gerar_formato_textual(
        matriz_tratada, novo_vetor_operadores, novo_vetor_b, vetor_coef_tratado, novas_variaveis, [], [], 0
    )
    print(resultado)

    ProblemaPL.quantidade_variaveis = len(novas_variaveis)
    ProblemaPL.vetor_variaveis=np.array(novas_variaveis)
    ProblemaPL.vetor_coeficientes=vetor_coef_tratado
    ProblemaPL.matriz_coeficientes=matriz_tratada
    ProblemaPL.vetor_b=novo_vetor_b
    ProblemaPL.vetor_operadores_novo=novo_vetor_operadores

    # Retorna um novo objeto ProblemaPL atualizado
    return ProblemaPL

def gerar_indices_ordenados(solucao_inicial, vetor_variaveis):
    # Converte o vetor de variáveis para lista, caso seja um array NumPy
    vetor_variaveis_lista = vetor_variaveis.tolist()
    # Encontra os índices das variáveis da solução inicial no vetor de variáveis
    indices = [vetor_variaveis_lista.index(var) for var in solucao_inicial]
    # Ordena os índices em ordem crescente
    indices.sort()
    return indices

def calcular_vetores_Cn_Cb(matriz_coeficientes, coef_objetivo, base_indices):
    num_linhas, num_colunas = matriz_coeficientes.shape
    non_base_indices = [i for i in range(num_colunas) if i not in base_indices]
    
    # Vetor Cb (coeficientes da função objetivo para as variáveis da base)
    Cb = coef_objetivo[base_indices]
    
    # Vetor Cn (coeficientes da função objetivo para as variáveis fora da base)
    Cn = coef_objetivo[non_base_indices]
    
    return Cn, Cb

def calcular_matrizes_B_N(matriz_coeficientes, base_indices):
    # Matrizes B e N
    B = matriz_coeficientes[:, base_indices]  # Colunas da base
    non_base_indices = [i for i in range(matriz_coeficientes.shape[1]) if i not in base_indices]
    N = matriz_coeficientes[:, non_base_indices]  # Colunas fora da base
    
    return B, N

def verificar_otimalidade(B, N, Cb, Cn):
    # Calcula B⁻¹ * N
    try:
        B_inv = np.linalg.inv(B)  # Inversa de B
        # print("Inversa da matriz B:\n", B_inv)
    except np.linalg.LinAlgError:
        return False, None  # B não é invertível, indicando um problema na base

    Cn_reduzido = Cn - np.dot(Cb @ B_inv, N)  # Fórmula Cn' - Cb * B⁻¹ * N

    # Se todos os valores de Cn_reduzido são >= 0, a solução é ótima
    return np.all(Cn_reduzido >= 0), Cn_reduzido

def regra_bland(delta):
    # Passo 2: Encontrar os índices para os quais delta < 0
    indices_negativos = np.where(delta < 0)[0]
    if len(indices_negativos) == 0:
        return None  # Nenhuma variável pode entrar na base, solução ótima encontrada
    # Passo 3: Se houver múltiplos, escolher a variável de entrada com a menor posição (menor índice)
    j_entrada = min(indices_negativos)  # Regra de Bland: primeiro menor índice
    return j_entrada

def calcular_d_B(B_inv, N, j):
    # Seleciona a j-ésima coluna de N
    Nj = N[:, j]  # Coluna j de N
    # Calcula d_B = -B_inv * Nj
    d_B = -np.dot(B_inv, Nj)
    # Verifica se todos os componentes de d_B são não negativos
    if np.all(d_B >= 0):
        print("Problema ilimitado")
        return None
    else:
        # print("Problema não ilimitado")
        return d_B

def calcular_x_B(B_inv, b):
    # Calcula x_B = B_inv * b
    x_B = np.dot(B_inv, b)
    return x_B

def calcular_t_star(x_B, d_B):
    # Filtra os componentes de d_B que são negativos
    indices_negativos = np.where(d_B < 0)[0]
    if len(indices_negativos) == 0:
        return False, False  # Caso não existam direções negativas, retorna infinito
    # Calcula t* como o mínimo de -x_B[i] / d_B[i] para os componentes negativos de d_B
    t_values = -x_B[indices_negativos] / d_B[indices_negativos]
    # print("t_values:", t_values)
    t_star = np.min(t_values)
    k = indices_negativos[np.argmin(t_values)]
    # print("k:", k)
    return k, t_star

def atualizar_B_N_C(j, k, B, N, C_B, C_N):    
    # Atualiza a matriz B (substitui a coluna k pela coluna j de N)
    B_atualizado = B.copy()
    B_atualizado[:, k] = N[:, j]
    
    # Atualiza a matriz N (substitui a coluna j por aquela de B na posição k)
    N_atualizado = N.copy()
    N_atualizado[:, j] = B[:, k]
    
    # Atualiza os vetores C_B e C_N (substitui a variável que sai pela que entra)
    C_B_atualizado = C_B.copy()
    C_B_atualizado[k] = C_N[j]
    
    C_N_atualizado = C_N.copy()
    C_N_atualizado[j] = C_B[k]
    
    return B_atualizado, N_atualizado, C_B_atualizado, C_N_atualizado

def atualizar_indices_base(indices_base, j_entrada, k_saida):
    indices_atualizados = indices_base.copy()  # Copia para evitar modificar o original
    indices_atualizados[k_saida] = j_entrada   # Substitui a variável que saiu pela nova
    return indices_atualizados

def obter_Cn_Cb_N_B(problema: ProblemaPL, colunas_base):
    Cn, Cb = calcular_vetores_Cn_Cb(problema.matriz_coeficientes, problema.vetor_coeficientes, colunas_base)
    solucao_0 = variaveis_basicas(colunas_base, problema.vetor_variaveis)
    B, N = calcular_matrizes_B_N(problema.matriz_coeficientes, colunas_base)
    # print("Vetor Cn:", Cn)
    # print("Vetor Cb:", Cb)
    # print("Matriz B:\n", B)
    # print("Matriz N:\n", N)
    xB = np.dot(np.linalg.inv(B), problema.vetor_b)
    solucao = np.dot(Cb.T, xB)
    # print("Colunas que formam a base:", solucao_0, " = ", xB)
    # print("Custo atual da solução: ", solucao)
    return B, N, Cb, Cn

def verificar_artificiais_na_base(B, variaveis_artificiais, B_inv, vetor_b):
    # Calcula os valores das variáveis básicas
    x_B = np.dot(B_inv, vetor_b)
    # print("x_B", x_B)
    # Identifica quais variáveis artificiais ainda estão na base
    artificiais_na_base = []
    # Vamos iterar sobre as variáveis artificiais
    for i, var in enumerate(variaveis_artificiais):
        # Verifica se a variável i (associada ao índice da variável artificial) está na base
        if i < B.shape[1]:  # Assegura que estamos dentro dos limites das colunas de B
            artificiais_na_base.append(i)
    # Verifica se não há variáveis artificiais na base
    # print("artificiais_na_base", artificiais_na_base)
    if not artificiais_na_base:
        print("Nenhuma variável artificial na base.\nO PL original é viável!")
        return "viável"
    # Verifica o valor das variáveis artificiais
    for indice_var in artificiais_na_base:
        valor_var = x_B[indice_var]  # Valor da variável artificial na solução básica
        if valor_var > 0:
            print(f"Problema inviável! A variável artificial {variaveis_artificiais[indice_var]} está na base com valor positivo ({valor_var}).")
            return "inviável"
    print("Variáveis artificiais estão na base, mas todas com valor zero. Podemos removê-las antes da Fase II.")
    return "remover"

def remover_artificiais(B, N, Cb, Cn, vetor_b, variaveis_artificiais):
    # Encontra as variáveis artificiais que ainda estão na base
    artificiais_na_base = [i for i, col in enumerate(B.T) if i in variaveis_artificiais]

    if not artificiais_na_base:
        print("Nenhuma variável artificial na base. Pronto para a Fase II.")
        return B, N, Cb, Cn, vetor_b  # Nenhuma mudança necessária

    print(f"Variáveis artificiais na base: {artificiais_na_base}. Iniciando remoção...")

    for i in artificiais_na_base:
        # Tenta encontrar uma variável não-básica para entrar no lugar da artificial
        for j, coluna_N in enumerate(N.T):
            if coluna_N[i] != 0:  # Se a coluna permitir pivotear
                print(f"Substituindo variável artificial {i} por variável {j}")
                
                # Troca a variável artificial pela variável não-básica correspondente
                B[:, i] = coluna_N
                Cb[i] = Cn[j]

                # Atualiza as matrizes removendo a variável artificial e ajustando o vetor_b
                N = np.delete(N, j, axis=1)  # Remove a variável escolhida de N
                Cn = np.delete(Cn, j)  # Remove seu custo de Cn
                
                break
    print("Remoção concluída. Pronto para a Fase II.")
    return B, N, Cb, Cn, vetor_b

def metodo_simplex(B, N, Cb, Cn, vetor_b, vetor_variaveis, colunas_base, iteracao):
    iteracao = iteracao + 1
    print("\n------------------------------------------------------------------------------------------------------------------------------------------------------------")
    
    print("\nIteração:", iteracao)
    otimalidade, custos_reduzidos = verificar_otimalidade(B, N, Cb, Cn)
    xB = np.dot(np.linalg.inv(B), vetor_b)
    solucao = np.dot(Cb.T, xB)
    print("Custo atual da solução: ", solucao)

    solucao = variaveis_basicas(colunas_base, vetor_variaveis)
    print("Colunas que formam a base:", solucao, " = ", xB)
    indices_artificiais = [i for i, var in enumerate(vetor_variaveis) if var.startswith("a_")]

    if otimalidade:
        B_inv = np.linalg.inv(B)  
        xB = np.dot(B_inv, vetor_b)
        Z_otimo = np.dot(Cb.T, xB)
        if indices_artificiais:
            print("\033[32mSolução ótima do PL extra:\033[0m", Z_otimo)
        else:
            print("\033[32mSolução ótima do PL original:\033[0m", Z_otimo)
        
        return Z_otimo, B, colunas_base, solucao
    else:
        print("\033[31mA solução não é ótima.\033[0m")
        solucao_0 = variaveis_basicas(colunas_base, vetor_variaveis)

        j_entrada = regra_bland(custos_reduzidos)
        print("Índice da coluna da matriz N que entrará na base:", j_entrada)
        B_inv = np.linalg.inv(B)
        x_B = calcular_x_B(B_inv, vetor_b)
        d_B = calcular_d_B(B_inv, N, j_entrada)
        k_saida, t_star = calcular_t_star(x_B, d_B)
        print("Índice da coluna da matriz B que sairá da base:", k_saida)

        B_atualizado, N_atualizado, C_B_atualizado, C_N_atualizado = atualizar_B_N_C(j_entrada, k_saida, B, N, Cb, Cn)
        B_inv = np.linalg.inv(B_atualizado)
        colunas_base = atualizar_indices_base(colunas_base, j_entrada, k_saida)

        return metodo_simplex(B_atualizado, N_atualizado, C_B_atualizado, C_N_atualizado, vetor_b, vetor_variaveis, colunas_base, iteracao)

def metodo_das_duas_fases(resultado, problema: ProblemaPL):
    resultado0, colunas_base, colunas_nao_encontradas = possui_solucao_basica_viavel(problema.matriz_coeficientes)
    # Salva os valores do PL original
    vetor_coeficientes_original = problema.vetor_coeficientes
    vetor_variaveis_original = problema.vetor_variaveis
    matriz_coeficientes_original = problema.matriz_coeficientes 

    print("\nNão há solução básica inicial viável visível. \nNecessita de um PL extra.\n")
    print(" PL extra: \n")
    matriz_tratada_aux, novas_variaveis, vetor_coef_tratado_aux, variaveis_artificiais = adicionar_variavel_artificial(problema.matriz_coeficientes, problema.vetor_variaveis, problema.vetor_operadores, resultado, problema.vetor_coeficientes, colunas_nao_encontradas)
    coef_obj_aux = vetor_coef_pl_auxiliar(novas_variaveis)
    resultado = gerar_formato_textual(matriz_tratada_aux, problema.vetor_operadores_novo, problema.vetor_b, coef_obj_aux, novas_variaveis, [], [], 0)
    print(resultado)

    ProblemaPL.vetor_variaveis = novas_variaveis
    ProblemaPL.vetor_coeficientes = coef_obj_aux
    ProblemaPL.matriz_coeficientes = matriz_tratada_aux
    ProblemaPL.vetor_b = problema.vetor_b
    PL_aux = ProblemaPL

    # resultado_original = gerar_formato_textual(problema.matriz_coeficientes, problema.vetor_operadores, problema.vetor_b, problema.vetor_coeficientes, problema.vetor_variaveis, [], [], 0)
    # print(resultado_original)
    resultado2, colunas_base, colunas_nao_encontradas = possui_solucao_basica_viavel(matriz_tratada_aux)
    B, N, Cb, Cn = obter_Cn_Cb_N_B(PL_aux, colunas_base)
    Z_otimo, B_aux, colunas_base, solucao = metodo_simplex(B, N, Cb, Cn, PL_aux.vetor_b, PL_aux.vetor_variaveis, colunas_base, 0)

    indices_artificiais = [i for i, var in enumerate(solucao) if var.startswith("a_")]
    if Z_otimo == 0:
        resultado = verificar_artificiais_na_base(B_aux, indices_artificiais, np.linalg.inv(B_aux), PL_aux.vetor_b)
        if resultado == "viável":
            print("\n------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("\nIniciando Fase II...")

            PL_aux.vetor_coeficientes = vetor_coeficientes_original
            PL_aux.vetor_variaveis = vetor_variaveis_original
            PL_aux.matriz_coeficientes = matriz_coeficientes_original

            B, N, Cb, Cn = obter_Cn_Cb_N_B(PL_aux, colunas_base)

            metodo_simplex(B, N, Cb, Cn, PL_aux.vetor_b, PL_aux.vetor_variaveis, colunas_base, 0)  # Chama a Fase II
        elif resultado == "remover":
            print("Removendo variáveis artificiais da base antes da Fase II...")
            B, N, Cb, Cn, vetor_b = remover_artificiais(B, N, Cb, Cn, PL_aux.vetor_b, indices_artificiais)
            PL_aux.vetor_coeficientes = vetor_coeficientes_original
            PL_aux.vetor_variaveis = vetor_variaveis_original
            PL_aux.matriz_coeficientes = matriz_coeficientes_original
            print("Iniciando Fase II do Simplex...")
            metodo_simplex(B, N, Cb, Cn, problema.vetor_b, problema.vetor_variaveis, colunas_base, 0)
        else:
            print("\n\n\033[31mPL original inviável!\033[0m")
            print("(Não possui solução sem auxílio de variáveis artificiais) \n\n")
            print("Fim do algoritmo: O problema não tem solução viável.")
    else: 
        print("\n\n\033[31mPL original inviável!\033[0m")
        print("(Não possui solução sem auxílio de variáveis artificiais) \n\n")
        print("Fim do algoritmo: O problema não tem solução viável.")
