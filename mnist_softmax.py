#Baixando os datasets MNIST
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Importando o tensorflow
import tensorflow as tf 

#Criando variável de entrada, utilizado quando o tensorflow executa um cáculo
#Queremos introduzir qualquer numero de imagens MNIST, achatados em um vetor 784-dimencional
#Montando um tensor 2-D de pontos flutuantes, com forma [Nenhuma, 784]
#Nenhuma significa dimensão de qualquer comprimento
x = tf.placeholder(tf.float32, [None, 784])

#Pesos e preconceitos para o modelo, 
#Variável é um tensor modificável que vive no gráfico de operações interagindo com o Tensorflow
#Para aplicações de aprendizagem de máquina, geralmente tem os parâmetros do modelo
# tf.Variable o valor inicial da variável: neste caso, inicializa-se W e b como tensores de zeros
#W tem forma [784, 10] porque queremos multiplicar os vetores de imagens 784-dimencionais por ele 
#para produzir vetores 10-dimensionais de provas para as classes de diferança
#b tem forma [10] para que possamos adicioná-lo à saída

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implementado o modelo. Leva apenas uma linha para defini-lo

y = tf.nn.softmax(tf.matmul(x, W) + b)

#Primeiro multiplicamos x por W com tf.matmul. Este é invertido a partir de quando multipliquei-los à 
#nossa equação, obtando Wx, como um pequeno truque para lidar com x sendo um tensor 2-D com múltiplas 
#entradas. Em seguida adcione b, e finalmente, aplicar tf.nn.softmax

#TREINAMENTO

#Para treinar o nosso modelo, precisamos definir o que significa para o modelo ser bom. Bem na verdade
#no aprendizado de máquina normalmente definem o que significa para o modelo ser ruim.
#Chamamos a isto o custo, ou perda, e representa quão longe nosso modelo é do nosso resultado desejado
#Tentamos minimizar esse erro, quanto menor a margem de erro. o melhor nosso modelo é

#Uma função muito comum, muito bom para determinar a perda de um modelo é chamado de "cross-entropia"
#Cross-entropia é medir o quão inificiente nossas previsões são para descrever a verdade
# Hy(y) = -Somatóriodei yi*log(yi)

#Para implementar cross-entropia é preciso adicionar um novo espaço resrvado para inserir as respostas corretas

y_ = tf.placeholder(tf.float32, [None, 10])

#Então podemos implementar a função cross-entropia

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

#Primeiro calcular tf.log, o logaritmo de cada elemento de y
#Multiplicar cada elemento de y_ com o elemento correspondente de tf.log(y)
#adiciona tf.reduce_sum os elementos na segunda dimensão de y, devido às reduction_indices=[1] parâmetros
#tf.reduce_mean calcula a média sobre todos os exemplos do lote.

#Agora que sabemos o que queremos que o nosso modelo faça, o Tensorflow sabe todo o gráfico de seus cálculos
#ele utiliza o algoritmo de retropropagação "BackPropagation" e gradiente descendente
#Retorna uma única operação que, quando executado, faz uma etapa de formação gradiente descendente, ajustando 
#ligeiramente suas variáveis para reduzir a perda

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Antes de iniciar o treinamento temos que criar uma operação para inicializar as variáveis que criamos

init = tf.initialize_all_variables()

#Lançando o modelo em uma sessão, e executar a operação que inicializa as variáveis

sess = tf.Session()
sess.run(init)

#Vamos executar o passo de formação 1000 vezes

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

#Cada etapa do circito, temos um lote de 100 pontos de dados aleatórios do nosso conjunto de treinamento 
#Alimentamos train_step nos dados de lotes para substituir os espaços reservados
#Usando pequenos lotes de dados aleatórios é chamado formação estocástica, neste caso estocástica gradiente descendente

#AVALIANDO NOSSO MODELO 

#Primeiro vamos descobrir onde previmos o rótulo correto. tf.argmax é uma função que retorna o índice de entrada mais
#alta em um tensor ao longo do eixo. Por exemplo tf.argmax(y, 1) é nosso modelo de rótulo mais provável para cada
#entrada, enquanto tf.argmax(y_, 1) é a etiqueta correta
#Podemos utilizar tf.equal para verificar se a nosso previsão corresponde à verdade

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#Isso retorna uma lista de booleanos, que são convertido para números de pontos flutuantes, e em seguida, retiara 
#a média. Por exemplo [true, false, true, true] se tornaria [1, 0, 1, 1], que se tornaria 0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Finalmente pedimos a nosso precisão em nosso dados de teste

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#Deve ser cerca de 92%





