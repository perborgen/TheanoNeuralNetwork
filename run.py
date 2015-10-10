import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
import pandas as pd

df = pd.read_csv('data.csv',header=0)
x = T.dvector()
y = T.dscalar()

Y_data = np.array(df["label"])
X_data = np.array(df[["grade1", "grade2"]])
X_train = X_data[:70]
Y_train = Y_data[:70]
X_test = X_data[70:]
Y_test = Y_data[70:]

theta1 = theano.shared(np.array(np.random.rand(3, 3), dtype=theano.config.floatX))
theta2 = theano.shared(np.array(np.random.rand(4, 1), dtype=theano.config.floatX))

def layer(x, w):
	b = np.array([1], dtype=theano.config.floatX)
	new_x = T.concatenate([x, b])
	m = T.dot(w.T, new_x)
	h = nnet.sigmoid(m)
	return h

def grad_desc(cost, theta):
	alpha = 0.1
	theta = theta - (alpha * T.grad(cost, wrt=theta))
	return theta


hid1 = layer(x,theta1)
out1 = T.sum(layer(hid1,theta2))
fc = (out1-y)**2

cost = theano.function(inputs=[x,y], outputs=fc, updates=[
	(theta1,grad_desc(fc, theta1)),
	(theta2,grad_desc(fc, theta2))
	])

forward = theano.function(inputs=[x],outputs=out1)

cur_cost = 0

def run(x,y):
	for i in range(2000):
		for j in range(len(x)):
			cur_cost = cost(x[j], y[j])
		if i % 500 == 0:
			print 'cost: ', cur_cost


def test(x_test,y_test):
	score = 0
	for k in xrange(len(y_test)):
		prediction = forward(x_test[k])
		prediction = round(prediction)
		if prediction == y_test[k]:
			score += 1
	percent = score / float(len(y_test))
	print 'Scored ', score, ' out of ', len(y_test)
	print 'Success rate: ', percent
	return percent

run(X_train,Y_train)
test(X_test,Y_test)