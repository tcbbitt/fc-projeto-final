# Projeto Final da disciplina de Ferramentas Computacionais Modernas para Computação Científica e Aprendizado de Máquina 2025/01 - PPGEEL - Thiago Carvalho Bittencourt

## Classificadores Binários para detecção de falha de rolamento utilizando o dataset público Ottawa 2023.

### Dataset
- O dataset utilizado foi o Ottawa 2023 (disponível em: https://data.mendeley.com/datasets/y2px5tg92h/5). A velocidade e carga constantes, foram testados 20 rolamentos com 3 características de funcionamento (Normalidade, desenvolvendo falha, falha) e 4 tipos de falha (inner race, outer race, ball e cage).Totalizando 60 aquisições de 10s cada, com frequência de amostragem de 42kHz.
- Para esse trabalho foram utilizados segmentos de 1s no treinamento e avaliação do modelo, de modo a aumentar o tamanho do dataset.
- No treino os trechos de 1s são selecionados aleatoriamente a partir do sinal de 10s. Já no teste e validação os sinais de 1s não tem sobreposição.
#### Transforms
Conforme a documentação do Pytorch, recomenda-se criar uma classe para cada possível transformação desejada, aplicando ao final do getitem da classe dataset. As transformações criadas foram ToTensor (necessária para o funcionamento do modelo), FFT (mudança da representação de entrada), Normalize (definição de normalização entry-wise) e random-gain (ganho aleatório no sinal). 
#### Splits
- De modo a evitar vazamento de dados, os splits são feitos com base no id do rolamento, ou seja, um rolamento que aparece no teste não pode aparecer na validação ou treino e vice-versa.
#### Wrapper
- Como o dataset é relativamente pequeno, no treinamento pode-se aumentar artificialmente o tamanho do dataset passando o mesmo sinal N vezes (hiperparâmetro), porém como o começo do trecho de 1s é aleatorizado a cada chamada, aumenta-se a diversidade no treino.
#### Dataloaders
- Definiu-se uma função auxiliar de modo que os dataloader pudessem ser chamados de maneira mais modular.

### Models
- Nesse trabalho não foram desenvolvidas arquiteturas, mas sim utilizou-se de uma arquitetura reconhecida na identificação de falhas a partir de sinais de vibração. a rede utilizada 'DCNN for one-dimensional signals in Pytorch' (disponível em:https://github.com/redone17/deep-sensor/tree/master) é uma rede convolucional 1D. É possível alterar o nível de complexidade da rede a partir do parâmetro DCNNXX. Nesse trabalho foram testadas a rede DCNN08(mais simples) e a DCNN19 (mais complexa disponível).
- Apesar do dataset ser multirótulo (4 falhas), esse trabalho limita-se a criação de classificadores binários para cada tipo de falha (identifica se existe a falha X) dada a complexidade do caso multirótulo ou multiclasse. A perda é a BCEWithLogitsLoss com otimizador Adam. Como métrica de avaliação, como no caso binário o conjunto é desbalanceado, escolhe-se a AUROC.

### Trainer
- Foram definidas as funções para treino e validação/teste para serem usadas no loop de treinamento e avaliação

### Resultados e comentários finais
 - Para o log de resultados, utiliza-se a plataforma wandb. Para cada modelo, tira-se a média de 5 seeds para avaliação com os melhores hiperparâmetros encontrados. O tensorboard foi utilizado apenas para a visualização da rede.