from src.desenvolvimento.model.model_pessoa import Pessoa
from src.desenvolvimento.Algoritimo_escolhido.Rodando_Algoritimo_escolhido import rodar_Algoritimo_escolhido
from fastapi import APIRouter

router = APIRouter()


@router.get('/inserir_dados')
def Inserir_dados():
    """
    :return: Inserir dados para análise.
    """

    pessoa = Pessoa(
        nome='Kadu',
        idade=34,
        genero=1,
        altura=1.65,
        peso=110,
        frequencia_alcool=0,
        alimentos_alto_teor_calorico=1,
        come_vegetais_nas_refeicoes=0,
        quantidade_refeicoes_dia=3,
        monitora_calorias_que_ingere=1,
        fuma=0,
        quantidade_agua_por_dia=2,
        membro_familiar_com_sobre_peso=0,
        frequencia_atividade_fisica=0,
        tempo_que_passa_dispositivos_tecnologicos=5,
        come_algo_entre_as_refeicoes=0,
        qual_transporte_costuma_usar=4
    )

    resultado = rodar_Algoritimo_escolhido(pessoa)

    return f'Olá {pessoa.exibir_nome()}, {pessoa.exibir_resultado(resultado)}'
