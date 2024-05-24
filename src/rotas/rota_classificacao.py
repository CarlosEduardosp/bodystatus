from src.model.model_pessoa import Pessoa
from src.Algoritimo_escolhido.Rodando_Algoritimo_escolhido import rodar_Algoritimo_escolhido

pessoa = Pessoa(
        nome='Kadu',
        idade=38,
        genero=0,
        altura=1.80,
        peso=75,
        frequencia_alcool=0,
        alimentos_alto_teor_calorico=0,
        come_vegetais_nas_refeicoes=0,
        quantidade_refeicoes_dia=3,
        monitora_calorias_que_ingere=0,
        fuma=0,
        quantidade_agua_por_dia=2,
        membro_familiar_com_sobre_peso=0,
        frequencia_atividade_fisica=4,
        tempo_que_passa_dispositivos_tecnologicos=0,
        come_algo_entre_as_refeicoes=1,
        qual_transporte_costuma_usar=4
    )

resultado = rodar_Algoritimo_escolhido(pessoa)

print(f'Olá {pessoa.exibir_nome()}, {pessoa.exibir_resultado(resultado)}')