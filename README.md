# Avaliação Sprint 2

para a execução do projeto foram utilizadas diversas bibliotecas, como a Openml, Pymongo e Sklearn
o codigo começa chamando uma classe responsavel por buscar o dataset através da biblioteca openml, em seguida os dados são separados para o processo de treino utilizando a biblioteca sklearn.
ao obter os resultados do teste e a acuracia, outra classe responsavel por manipular o banco é chamada, para enviar ao banco o dataset e as informações obtidas pelo teste
e logo em seguida ja chama outra função dessa classe para mostrar os resultados.
