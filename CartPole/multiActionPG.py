''' Based on https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 '''
import gym
import numpy as np
import random

# Parameters
H = 200 # number of hidden layer neurons
batch_side = 10 # episodes per param update
learning_rate = 1e-2
gamma = 0.99 # decay factor
decay_rate = 0.9 # decay factor for RMSProp leaky sum of grad^2
batch_size = 128
num_actions = 2

# Model initialization
D = 8 # input dimensions
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(num_actions,H) / np.sqrt(H*num_actions)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() }
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(f):
    f_exp = np.exp(f)
    f_softmax /= np.sum(f_exp)
    return f_exp

def discount_rewards(reward_sum):
    discounted_r = np.ones(reward_sum)
    discounted_r *= reward_sum
    running_decay = 0
    for t in xrange(int(reward_sum)):
        discounted_r[t] *= gamma ** t
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0
    f = np.dot(model['W2'], h) # log(p1) - log(p2)
    p = softmax(f)
    return p, h

def policy_backward(eph, epdf, epx):
    dW2 = np.dot(df, eph.T)
    dh = np.dot(model['W2'].T, epdf)
    dh[eph <= 0] = 0 # backprop relu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

def one_hot(y, length):
    v = np.ones(length)
    v[y] += 1
    return v


env = gym.make('CartPole-v0')
observation = env.reset()
prev_x = np.zeros(4)
xs, hs, df = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

batch_rewards = []

while True:
    # env.render()

    # set up input
    current_x = np.array(observation)
    velocity_x = current_x - prev_x
    x = np.concatenate((current_x, velocity_x))
    prev_x = current_x

    # choose action
    # y is the index of the chosen action
    prob, h = policy_forward(x)
    y_value = np.random.uniform()
    for i in range(len(prob)):
        if np.sum(prob[:i+1]) > y_value: y = i

    xs.append(x)
    hs.append(h)

    df = -prob + one_hot(y, num_actions)
    dfs.append(df)

    observation, reward, done, info = env.step(y)
    reward_sum += reward

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdf = np.vstack(dfs)
        xs, hs, dfs = [], [], []

        discounted_r = discount_rewards(reward_sum)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        #discounted_r = discounted_r / 100.0 - 1.0

        epdf = epdf * discounted_r
        grad = policy_backward(eph, epdf, epx)
        for k in model: grad_buffer[k] += grad[k]

        batch_rewards.append(reward_sum)

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            print 'episode %d finished with reward average %d' % (episode_number, np.mean(np.array(batch_rewards)))
            print np.max(batch_rewards)
            batch_rewards = []

        reward_sum = 0
        observation = env.reset()
        prev_x = np.zeros(4)


