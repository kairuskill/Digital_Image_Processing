#importar tensorflow
import tensorflow as tf

#declarar as variáveis de entrada (x, y)
x = tf.constant(3, dtype=tf.float16)
y = tf.constant(7, dtype=tf.float16)

#definir as operações do grafo 
a = tf.add(x,y)
b = tf.multiply(x, y)
c = tf.subtract(a, b)
d = tf.add(c, x)

#iniciar a sessão do tf
sess = tf.Session()

#armazenar o output em uma variável 
output = sess.run(d)
print("Output: {}".format(output))

#fechar a sessão 
sess.close()
