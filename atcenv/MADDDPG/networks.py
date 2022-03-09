import numpy as np
import tensorflow as tf

HIDDEN1_UNITS_ = 60
HIDDEN2_UNITS_ = 70
HUBER_LOSS_DELTA = 1

TAU = 0.001  # Target Network HyperParameters, for soft update of target parameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic
EPSILON = 0.1  
ALPHA = 0.9  # learning rate
GAMMA = 0.99

def huber_loss(y_true, y_pred):
    from keras import backend as K
    err = y_true - y_pred

    cond = K.abs(err) <= HUBER_LOSS_DELTA
    if cond == True:
        loss = 0.5 * K.square(err)

    else:
        loss = 0.5 * HUBER_LOSS_DELTA ** 2 + HUBER_LOSS_DELTA * (K.abs(err) - HUBER_LOSS_DELTA)

    return K.mean(loss)


class Critic(object):
    def __init__(self, name, sess, state_size, action_size):
        self.sess = sess
        self.action_size = action_size
        self.net_name = name

        tf.compat.v1.keras.backend.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.compat.v1.global_variables_initializer())


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = TAU * critic_weights[i] + (1 - TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):

        state_input = tf.keras.layers.Input(shape=[state_size])
        state_h1 = tf.keras.layers.Dense(HIDDEN1_UNITS_, activation='relu')(state_input)
        state_h2 = tf.keras.layers.Dense(HIDDEN2_UNITS_, activation='linear')(state_h1)

        action_input = tf.keras.layers.Input(shape=[action_dim])
        action_h1 = tf.keras.layers.Dense(HIDDEN2_UNITS_, activation='linear')(action_input)

        #layer in the middle that merges the two before
        merged = tf.keras.layers.add([state_h2, action_h1])
        merged_h1 = tf.keras.layers.Dense(HIDDEN2_UNITS_, activation='relu')(merged)

        output = tf.keras.layers.Dense(action_dim, activation='linear')(merged_h1)

        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)

        self.opt = tf.keras.optimizers.RMSprop(lr=LRC)

        # If the model has multiple outputs, you can use a different loss
        # on each output by passing a dictionary or a list of losses.
        # The loss value that will be minimized by the model
        # will then be the sum of all individual losses.
        model.compile(loss=huber_loss, optimizer=self.opt)  # 'mse'

        return model, action_input, state_input


class Actor(object):
    def __init__(self, name, sess, state_size, action_size):
        self.sess = sess
        self.TAU = TAU
        self.net_name = name

        # Everything in TensorFlow is represented as a computational graph that consists of nodes and edges,
        # where nodes are the mathematical operations, and edges are the tensors.
        # in order to execute a graph, we need to initialize a TensorFlow session as follows:
        tf.compat.v1.keras.backend.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size)

        # placeholders are variables where you only define the type and dimension but will not assign the value
        tf.compat.v1.disable_eager_execution()
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(LRA).apply_gradients(grads)

        # tf.global_variables_initializer() allocates resources for the variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def compile(self):
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size):

        # input layer
        state_input = tf.keras.layers.Input(shape=[state_size])

        # densely-connected NN layer
        # activation function: relu -  rectified linear unit (ReLU)
        # dense layer - Dense layer: A linear operation in which every input is connected to every output by a weight
        # (so there are n_inputs * n_outputs weights - which can be a lot!). Generally followed by a non-linear activation function
        h0 = tf.keras.layers.Dense(HIDDEN1_UNITS_, activation='relu')(state_input)
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS_, activation='relu')(h0)
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS_, activation='relu')(h1)

        init1 = tf.keras.initializers.RandomNormal(mean=0, stddev=1 / np.sqrt(state_size), seed=20)
        #act1 = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=init1)(h1)
        #act2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=init1)(h1)
        act1 = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=init1)(h1)
        act2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=init1)(h1)

        output = tf.keras.layers.concatenate([act1, act2])
        model = tf.keras.Model(inputs=state_input, outputs=output)

        return model, model.trainable_weights, state_input