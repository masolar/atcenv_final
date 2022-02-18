import tensorflow as tf

HIDDEN1_UNITS = 120
HIDDEN2_UNITS = 140

HUBER_LOSS_DELTA = 1


def huber_loss(y_true, y_pred):
    from keras import backend as K
    err = y_true - y_pred

    cond = K.abs(err) <= HUBER_LOSS_DELTA
    if cond == True:
        loss = 0.5 * K.square(err)

    else:
        loss = 0.5 * HUBER_LOSS_DELTA ** 2 + HUBER_LOSS_DELTA * (K.abs(err) - HUBER_LOSS_DELTA)

    return K.mean(loss)

# the critic network is the value function. the critic produces a Temporal-Difference error signal
# given the state and resultant reward
class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

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
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):

        state_input = tf.keras.layers.Input(shape=[state_size])
        state_h1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='relu')(state_input)
        state_h2 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='linear')(state_h1)

        action_input = tf.keras.layers.Input(shape=[action_dim])
        action_h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='linear')(action_input)

        #layer in the middle that merges the two before
        merged = tf.keras.layers.add([state_h2, action_h1])
        merged_h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu')(merged)

        output = tf.keras.layers.Dense(action_dim, activation='linear')(merged_h1)

        model = tf.keras.Model(inputs=[state_input, action_input], outputs=output)

        self.opt = tf.keras.optimizers.RMSprop(lr=self.LEARNING_RATE)

        # If the model has multiple outputs, you can use a different loss
        # on each output by passing a dictionary or a list of losses.
        # The loss value that will be minimized by the model
        # will then be the sum of all individual losses.
        model.compile(loss=huber_loss, optimizer=self.opt)  # 'mse'

        #plot_model(model, to_file='DDPG_Critic_model.png', show_shapes=True)

        return model, action_input, state_input
