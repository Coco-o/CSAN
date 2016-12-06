from __future__ import division
from __future__ import print_function
import numpy as np
from util import *
from classifiers import WeightedBinaryNN #fake classifier
import sys


class AdvSampler(object):
    '''
    The full model, containing three classifiers
    '''
    def __init__(self, config):
        self.sampler = WeightedBinaryNN(config['sampler_parameter'])
        self.adv_accept = WeightedBinaryNN(config['adv_acc_parameter'])
        self.adv_reject = WeightedBinaryNN(config['adv_rej_parameter'])
        self.sampler.random_init()
        self.adv_accept.random_init()
        self.adv_reject.random_init()

        self.adv_acc_step_num = config['adv_acc_step_num']
        self.adv_acc_learning_rate = config['adv_acc_learning_rate']
        self.adv_rej_step_num = config['adv_rej_step_num']
        self.adv_rej_learning_rate = config['adv_rej_learning_rate']
        self.sampler_step_num = config['sampler_step_num']
        self.sampler_learning_rate = config['sampler_learning_rate']
        self.coeff = config['coeff']
        self.sgd = config['sgd']
        self.sampler_batch_size = config['sampler_batch_size']
        self.pred_s = 0 # store preditions for source data
        self.pred_a = 0
        self.pred_r = 0
        self.update_pred_freq = config['update_pred_freq']
        #self.flags = 0 # 0 means source data, 1 means target data
        #self.batch_size

    def update_sampler(self, x):
        '''
        minibatch implementation, update 1 epoch
        '''
        batch_size = int(x.shape[0] * self.sampler_batch_size // 1)
        rounds = int(x.shape[0] // batch_size)
        update_rounds = x.shape[0] // (self.update_pred_freq * batch_size)
        #print(update_rounds)
        for i in range(rounds):
            if i != 0 and i % update_rounds == 0:
                self.pred_s = self.sampler.predict(x)
            st, ed = i * batch_size, i * batch_size + batch_size
            self.update_sampler_batch(x[st:ed, :], st, ed)

    def update_sampler_batch(self, x, st, ed):
        '''
        f1 = weighted_average(accept_prob, log(1-adv_acc_pred))
        f2 = weighted_average(rej_prob, log(1-adv_rej_pred))
        minimize f1 - coeff * f2 by gradient descent (only one step)
        small eps to smooth log(1 - g(x)) by log(1 + eps - g(x))
        '''
        eps = 10 ** (-8)
        n = x.shape[0]
        accept_prob = self.sampler.predict(x)
        #self.pred_s[st:ed, :] = accept_prob # update predictions
        reject_prob = 1 - accept_prob
        # first compute gradient wrt accetance probabilities
        '''
        adv_acc_pred, adv_rej_pred = self.adv_accept.predict(x), self.adv_reject.predict(x)
        loglike_acc, loglike_rej = np.log(1 + eps - adv_acc_pred), np.log(1 + eps - adv_rej_pred)
        part_1 = loglike_acc / (np.sum(accept_prob) + eps)
        part_2 = loglike_rej / (np.sum(reject_prob) + eps)
        part_3 = np.ones_like(part_1) * np.dot(accept_prob.T, loglike_acc) / (np.sum(accept_prob)**2 + eps)
        part_4 = np.ones_like(part_1) * np.dot(reject_prob.T, loglike_rej) / (np.sum(reject_prob)**2 + eps)
        '''
        z_pos = np.sum(self.pred_s) / self.pred_s.shape[0] # normalize factor for acc
        z_neg = 1 - z_pos # normalize factor for ref
        loglike_acc, loglike_rej = np.log(1 + eps - self.pred_a), np.log(1 + eps - self.pred_r)
        part_1 = loglike_acc[st:ed, :] / (z_pos + eps)
        part_2 = loglike_rej[st:ed, :] / (z_neg + eps)
        part_3 = np.ones_like(part_1) * np.dot(self.pred_s.T, loglike_acc) / (z_pos**2 * self.pred_s.shape[0] + eps)
        part_4 = np.ones_like(part_1) * np.dot(1 - self.pred_s.T, loglike_rej) / (z_neg**2 * self.pred_s.shape[0] + eps)

        grad_wrt_pred = part_1 - part_3 + self.coeff * (part_2 - part_4)
        self.sampler.update_with_grad_on_pred(x, grad_wrt_pred, self.sampler_learning_rate)



    def train(self, x_source, x_target, step_num):
        '''
        batch training
        x_source and x_target are datasize * dimension
        Iterative training:
            1. compute acceptance probabilities beta(x) for all x in the source dataset
            2. update two classifiers by doing gradient descent for some steps (adv_step_num)
            3. update sampler
        label: source-0, target-1, which is different from what is in the writing
        '''
        eps = 10 ** (-8)
        log = []
        n_s = x_source.shape[0]
        n_t = x_target.shape[0]
        # shuffle x_source
        idx_source = shuffle_index(x_source)
        x_source = x_source[idx_source, :]
        # shuffle both
        x = np.vstack((x_source, x_target))
        y = np.vstack((np.zeros((n_s, 1)), np.ones((n_t, 1)))).astype(int)
        idx = shuffle_index(x)

        x = x[idx, :]
        y = y[idx, :]
        #self.pred_s = np.zeros((n_s, 1))
        #self.pred_a = np.zeros((n_s, 1))
        #self.pred_r = np.zeros((n_s, 1))
        #self.flags = np.zeros(x.shape[0]).astype(int)
        #self.flags[idx >= ns] = 1

        for i in range(step_num):
            #sys.stdout.write('.')
            #sys.stdout.flush()
            #compute weight vector for x_source
            accept_prob = self.sampler.predict(x_source)
            self.pred_s = accept_prob
            #concatenate weight vectors
            weight_src_acc = accept_prob / np.sum(accept_prob)
            weight_src_rej = (1 - accept_prob) / np.sum(1 - accept_prob)
            '''
            balanced two classes
            '''
            weight_target = np.ones((n_t, 1)) / n_t
            weight_acc = np.vstack((weight_src_acc, weight_target))[idx, :]
            weight_rej = np.vstack((weight_src_rej, weight_target))[idx, :]
            '''
            imbalanced two classes
            '''
            #weight_acc = np.vstack((accept_prob, np.ones((n_t, 1))))
            #weight_rej = np.vstack((1 - accept_prob, np.ones((n_t, 1))))
            # train two discriminators:
            if self.sgd:
                batch_size = 1
            else:
                batch_size = x.shape[0]
            for j in range(self.adv_acc_step_num):
                # x.shape[0] -> batch gradient, 1 -> sgd
                self.adv_accept.train_epoch(x, y, weight_acc, self.adv_acc_learning_rate, batch_size)
            for j in range(self.adv_rej_step_num):
                self.adv_reject.train_epoch(x, y, weight_rej, self.adv_rej_learning_rate, batch_size)

            # print information:
            pred_sampler = self.get_result(x_source, 'sampler') #weights on source
            pred_acc_s, pred_acc_t = self.get_result(x_source, 'adv_acc'), self.get_result(x_target, 'adv_acc')
            pred_rej_s, pred_rej_t = self.get_result(x_source, 'adv_rej'), self.get_result(x_target, 'adv_rej')
            loss_acc = - np.sum(pred_sampler * np.log(1 + eps - pred_acc_s)) / np.sum(pred_sampler)\
                - np.sum(np.log(pred_acc_t + eps)) / n_t

            loss_rej = - np.sum((1 - pred_sampler) * np.log(1 + eps - pred_rej_s)) / np.sum(1 - pred_sampler)\
                - np.sum(np.log(pred_rej_t + eps)) / n_t

            self.pred_a = pred_acc_s
            self.pred_r = pred_rej_s

            # update sampler
            for j in range(self.sampler_step_num):
                self.update_sampler(x_source)
            pred_sampler, pred_sampler_t = self.get_result(x_source, 'sampler'), self.get_result(x_target, 'sampler')
            loss_sampler = - np.sum((1 - pred_sampler) * np.log(1 + eps - pred_sampler)) / np.sum(1 - pred_sampler)\
                - np.sum(np.log(pred_sampler_t + eps)) / n_t
            print(i+1, loss_acc, loss_rej, loss_sampler)
            log.append([i+1, loss_acc, loss_rej])
        return log


        #print()
    def get_result(self, x, model):
        model_dict = {
            'sampler': self.sampler,
            'adv_acc': self.adv_accept,
            'adv_rej': self.adv_reject
        }
        return model_dict[model].predict(x)

class WeightedLR(object):
    def __init__(self, dim):
        self.w = np.zeros((dim,1))
        self.b = 0

    def train(self, x, y, p, learning_rate):
        '''
        minimize 1/2(sum_i p_i) sum_i p_i (y-(wx_i+b))^2
        '''
        n = np.sum(p)
        eps = 10 ** (-8)
        max_step = 10000
        for i in range(max_step):
            pred = np.dot(x, self.w) + np.ones((x.shape[0], 1)) * self.b
            loss = np.sum(p * (pred-y)**2) / (2*n)
            #print(i, ': loss ', loss)
            grad_w = np.dot(x.T, (pred - y) * p) / (2*n)
            self.w -= learning_rate * grad_w
            grad_b = np.dot(p.T, pred-y) / (2*n)
            self.b -= learning_rate * grad_b[0,0]
            if np.sum(grad_w**2) + grad_b**2 <= eps:
                break
    def get_loss(self, x, y, p):
        n = np.sum(p)
        pred = np.dot(x, self.w) + np.ones((n, 1)) * self.b
        loss = np.sum(p * (pred-y)**2) / (2*n)
        return loss
    def predict(self, x):
        return np.dot(x, self.w) + np.ones((x.shape[0], 1)) * self.b
