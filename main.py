import funcoes

# Entrada
arquivo = "arquivo.txt"

ProblemaPL = funcoes.gerar_formato_matricial(arquivo)
ProblemaPL_forma_padrao = funcoes.transformar_para_forma_padrao(ProblemaPL, arquivo)

resultado, colunas_base, colunas_nao_encontradas = funcoes.possui_solucao_basica_viavel(ProblemaPL_forma_padrao.matriz_coeficientes)
if resultado:
    print("\nNão precisa de PL extra.\n")
    B, N, Cb, Cn = funcoes.obter_Cn_Cb_N_B(ProblemaPL_forma_padrao, colunas_base)
    funcoes.metodo_simplex(B, N, Cb, Cn, ProblemaPL_forma_padrao.vetor_b, ProblemaPL_forma_padrao.vetor_variaveis, colunas_base, 0)
else:
    funcoes.metodo_das_duas_fases(resultado, ProblemaPL_forma_padrao)
