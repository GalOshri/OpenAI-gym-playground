''' Based on Andrej Karpathy's https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 '''
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

# Model initialization
D = 8 # input dimensions
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() }
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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
    logp = np.dot(model['W2'], h) # log(p1) - log(p2)
    p = sigmoid(logp)
    return p, h

def policy_backward(eph, rdlogp, epx):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backprop relu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}


env = gym.make('CartPole-v0')
observation = env.reset()
prev_x = np.zeros(4)
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

batch_rewards = []

while True:
    env.render()

    # set up input
    current_x = np.array(observation)
    velocity_x = current_x - prev_x
    x = np.concatenate((current_x, velocity_x))
    prev_x = current_x

    # choose action
    aprob, h = policy_forward(x)
    y = 1 if np.random.uniform() < aprob else 0
    xs.append(x)
    hs.append(h)
    dlogps.append(y - aprob)

    observation, reward, done, info = env.step(y)
    reward_sum += reward

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        xs, hs, dlogps, drs = [], [], [], []


        discounted_r = discount_rewards(reward_sum)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        #discounted_r = discounted_r / 100.0 - 1.0

        epdlogp = epdlogp.squeeze() * discounted_r
        grad = policy_backward(eph, epdlogp, epx)
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


