# Avaliação Sprint 2

##Objetivo
 - O objetivo do projeto era obter um dataset e treinar um algoritmo que conseguisse prever os resultados de amostras semelhantes.

##Tecnologias utilizadas
 - Para a execução do projeto foram utilizadas diversas bibliotecas, principalmente a Openml, Pymongo e Sklearn, sendo que a Matplotlib e seaborn também foram utilizadas durante o desenvolvimento, mas não tem papeis grandes na versão final

##Descrição do funcionamento 
 - O codigo começa chamando uma classe responsavel por buscar o dataset através da biblioteca openml, em seguida os dados são separados para o processo de treino utilizando a biblioteca sklearn. Ao obter os resultados do teste e a acuracia, outra classe responsavel por manipular o banco é chamada, para enviar ao banco o dataset e as informações obtidas pelo teste e logo em seguida ja chama outra função dessa classe para mostrar os resultados.
