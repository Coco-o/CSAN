from __future__ import division
from __future__ import print_function
import numpy as np
from util import *
from classifiers import WeightedBinaryNN #fake classifier



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

        self.adv_step_num = config['adv_step_num']
        self.adv_learning_rate = config['adv_learning_rate']
        self.sampler_step_num = config['sampler_step_num']
        self.sampler_learning_rate = config['sampler_learning_rate']
        self.coeff = config['coeff']
        self.sgd = config['sgd']
        #self.batch_size

    def update_sampler(self, x):
        '''
        f1 = weighted_average(accept_prob, log(1-adv_acc_pred))
        f2 = weighted_average(rej_prob, log(1-adv_rej_pred))
        minimize f1 - coeff * f2 by gradient descent (only one step)
        small eps to smooth log(1 - g(x)) by log(1 + eps - g(x))
        '''
        eps = 0.001
        n = x.shape[0]
        accept_prob = self.sampler.predict(x)
        reject_prob = 1 - accept_prob
        # first compute gradient wrt accetance probabilities
        adv_acc_pred, adv_rej_pred = self.adv_accept.predict(x), self.adv_reject.predict(x)
        loglike_acc, loglike_rej = np.log(1 + eps - adv_acc_pred), np.log(1 + eps - adv_rej_pred)
        part_1 = loglike_acc / np.sum(accept_prob)
        part_2 = loglike_rej / np.sum(reject_prob)
        part_3 = np.ones_like(part_1) * np.dot(accept_prob.T, loglike_acc) / (np.sum(accept_prob)**2)
        part_4 = np.ones_like(part_1) * np.dot(reject_prob.T, loglike_rej) / (np.sum(reject_prob)**2)
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
        TODO: mini-batch
        '''
        n_s = x_source.shape[0]
        n_t = x_target.shape[0]
        x = np.vstack((x_source, x_target))
        y = np.vstack((np.zeros((n_s, 1)), np.ones((n_t, 1)))).astype(int)
        idx = shuffle_index(x)
        x = x[idx, :]
        y = y[idx, :]
        for i in range(step_num):
            print('.', end='')
            #compute weight vector for x_source
            accept_prob = self.sampler.predict(x_source)
            #concatenate weight vectors
            weight_src_acc = accept_prob / np.sum(accept_prob)
            weight_src_rej = (1 - accept_prob) / np.sum(1 - accept_prob)
            '''
            balanced two classes
            '''
            weight_target = np.ones((n_t, 1)) / n_t
            weight_acc = np.vstack((weight_src_acc, weight_target))
            weight_rej = np.vstack((weight_src_rej, weight_target))
            '''
            imbalanced two classes
            '''
            #weight_acc = np.vstack((accept_prob, np.ones((n_t, 1))))
            #weight_rej = np.vstack((1 - accept_prob, np.ones((n_t, 1))))
            # train two discriminators:
            for j in range(self.adv_step_num):
                # x.shape[0] -> batch gradient, 1 -> sgd
                if self.sgd:
                    batch_size = 1
                else:
                    batch_size = x.shape[0]
                self.adv_accept.train_epoch(x, y, weight_acc, self.adv_learning_rate, batch_size)
                self.adv_reject.train_epoch(x, y, weight_rej, self.adv_learning_rate, batch_size)
            # update sampler
            for j in range(self.sampler_step_num):
                self.update_sampler(x_source)
            # print information:
        print()
    def get_result(self, x, model):
        model_dict = {
            'sampler': self.sampler,
            'adv_acc': self.adv_accept,
            'adv_rej': self.adv_reject
        }
        return model_dict[model].predict(x)
