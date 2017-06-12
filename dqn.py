
import random
import gym
import numpy as np
from keras.models import Model, Input
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import losses


class Transition(object):
    def __init__(self, sequence, action, observation, reward, done):
        self.sequence = sequence
        self.action = action
        self.observation = observation
        self.reward = reward
        self.done = done


class Agent(object):
    MINIBATCH_SIZE = 32
    REPLAY_MEMORY_SIZE = 1e6
    AGENT_HISTORY_LENGTH = 4
    TARGET_NETWORK_UPDATE_FREQUENCY = 1e4
    DISCOUNT_FACTOR = 0.99
    ACTION_REPEAT = 4
    UPDATE_FREQUENCY = 4
    LEARNING_RATE = 0.000025
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.1
    FINAL_EXPLORATION_FRAME = 1e6
    REPLAY_START_SIZE = 50000
    IMG_WIDTH = 48
    IMG_HEIGHT = 48
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    NUM_OF_ACTIONS = 4

    def __init__(self):
        self.Q = self.get_model()
        self.Q_ = self.copy_model(self.Q)
        self.transitions = []
        self.train_minibatches = []
        self.exploration_factor = self.INITIAL_EXPLORATION
        self.env = gym.make("Breakout-v4")
        self.steps = 0


    def get_predicted_action(self, sequence):
        prediction = self.Q.predict(np.asarray([np.stack(sequence, -1)]))[0]
        return np.argmax(prediction)


    def copy_model(self, model):
        model.save_weights('weights.h5')
        new_model = self.get_model()
        new_model.load_weights('weights.h5')
        return new_model


    def get_action(self, sequence):
        n = random.random()
        if n <= self.exploration_factor:
            return self.env.action_space.sample()
        return self.get_predicted_action(sequence)


    def run(self, episodes, render=False, do_train=True):
        for episode in xrange(episodes):
            observation = self.env.reset()
            observation = self.pre_process_observation(observation)
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False

            action = 0
            while not done:
                if render:
                    self.env.render()
                if self.steps % self.ACTION_REPEAT == 0:
                    action = self.get_action(sequence)
                observation, reward, done, info = self.env.step(action)
                observation = self.pre_process_observation(observation)
                self.transitions.append(Transition(sequence, action, observation, reward, done))
                if len(self.transitions) > self.REPLAY_MEMORY_SIZE:
                    self.transitions = self.transitions[1:]
                sequence = sequence[1:]
                sequence.append(observation)

                self.steps += 1
                self.exploration_factor = max(self.FINAL_EXPLORATION, self.exploration_factor - 1.0/self.FINAL_EXPLORATION_FRAME)
                if len(self.transitions) >= self.REPLAY_START_SIZE:
                    self.train_minibatches.append(random.sample(self.transitions, self.MINIBATCH_SIZE))
                    if len(self.train_minibatches) >= self.UPDATE_FREQUENCY and do_train:
                        self.train()
                        self.train_minibatches = []

                if self.steps % self.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    self.Q_ = self.copy_model(self.Q)
                    print "############", self.exploration_factor


    def train(self):
        for minibatch in self.train_minibatches:
            sequences = []
            next_sequences = []
            for transition in minibatch:
                sequences.append(np.stack(transition.sequence, -1))
                next_sequences.append(np.stack(transition.sequence[1:] + [transition.observation], -1))

            targets = self.Q.predict(np.asarray(sequences))
            new_sequence_predictions = self.Q_.predict(np.asarray(next_sequences))

            for index, prediction in enumerate(new_sequence_predictions):
                done = minibatch[index].done
                reward = minibatch[index].reward
                action = minibatch[index].action
                max_q = np.max(prediction) if not done else 0
                targets[index][action] = reward + self.DISCOUNT_FACTOR * max_q

            print self.Q.train_on_batch(np.asarray(sequences), targets)


    def pre_process_observation(self, observation):
        observation = observation[32:-17, 8:-8]
        img = image.array_to_img(observation, 'channels_last')
        img = img.convert('L')
        img = img.resize(self.IMG_SIZE)
        observation = image.img_to_array(img, 'channels_last')
        return np.squeeze(observation)


    def get_model(self):
        input_layer = Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, self.AGENT_HISTORY_LENGTH))
        layer = BatchNormalization()(input_layer)
        layer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', data_format="channels_last", padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', data_format="channels_last", padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', data_format="channels_last", padding='same')(layer)
        layer = Flatten()(layer)
        layer = BatchNormalization()(layer)
        layer = Dense(512, activation='relu')(layer)
        layer = BatchNormalization()(layer)
        output_layer = Dense(self.NUM_OF_ACTIONS)(layer)
        model = Model(input_layer, output_layer)
        model.compile(Adam(self.LEARNING_RATE), loss=losses.mean_squared_error, metrics=['accuracy'])
        return model
