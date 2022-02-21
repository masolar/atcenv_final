import tensorflow as tf
import numpy as np

# HIDDEN1_UNITS_ = 120
# HIDDEN2_UNITS_ = 140
HIDDEN1_UNITS_ = 60
HIDDEN2_UNITS_ = 70

# the actor is the policy function. the actor produces an action given the current
# state of the environment.
class ActorNetwork(object):

    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        # Everything in TensorFlow is represented as a computational graph that consists of nodes and edges,
        # where nodes are the mathematical operations, and edges are the tensors.
        # in order to execute a graph, we need to initialize a TensorFlow session as follows:
        tf.compat.v1.keras.backend.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size)

        # placeholders are variables where you only define the type and dimension but will not assign the value
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

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
        act1 = tf.keras.layers.Dense(2, activation='sigmoid', kernel_initializer=init1)(h1)

        #output = tf.keras.layers.concatenate([act1, act2])
        model = tf.keras.Model(inputs=state_input, outputs=act1)

        return model, model.trainable_weights, state_input