# Trabalho 1 de visão computacional 
Primeiro trabalho na matéria de Visão Computacional.
O trabalho consiste em criar uma aplicação que possibilita
a visualização de um **objeto** (representado por um cubo) e uma 
**câmera** (representado por um referencial) em um ambiente tridimensional. 
Além disso, deve-se permitir que a câmera sofra transformações
de movimento rígido, ou seja, translação e rotação. E por útlimo,
permitir a visualização da projeção gerada pela câmera, podendo
ser alterado os parâmetros intrísecos dela.


**Dupla:** 
- Igor Mattos dos Santos Varejão
- Otavio Ferreira Cozer

### Preparação:
#### Automático
Para facilitar criamos um arquivo que cuida dessa etapa,
para executar basta:
1. Dar permissão ao arquivo ser executado
```bash
chmod u+x prepare_env.sh
```
2. E executá-lo:
```bash
./prepare_env.sh
```
#### Manual
Caso não funcione o automático, ou seja da sua preferência,
esses são os passos necessários para preparação do ambiente de execução:

Primeiro precisamos prepar o ambiente de execução:
1. Crie um ambiente virtual (`venv`):
```bash 
python3 -m venv t1_env
```
2. Agora ative o ambiente:
```bash
source activate t1_env/bin/activate
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```
### Execução
Caso a preparação do ambiente tenha ocorrido com sucesso,
basta executar o comando abaixo em seu terminal, estando
no diretório da atividade, que irá abrir uma página em 
seu navegador da internet com o trabalho.
```bash
streamlit run src/T1.py
```