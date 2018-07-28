"""
This parallelized singleplayer OpenAI gym MCTS code is adapted from the single-thread multiplayer board-game MCTS implementation
by Surag Nair at https://web.stanford.edu/~surag/posts/alphazero.html
"""

import math
import numpy as np
import itertools
import logging

EPS = 1e-8

logger = logging.getLogger(__name__)


def getBatchMCTSActionProbs(mcts_instances, states, envs, observations, reward_discount_factor, thread_pool, temps=None):
    if temps is None:
        temps = [1 for _ in states]

    results = thread_pool.map(lambda x: x[0].getActionProb(*x[1:]),
                                  zip(mcts_instances, states, envs, observations, itertools.repeat(reward_discount_factor),
                                      temps))

    probs_batch = np.asarray([x[0] for x in results])
    avg_value_batch = np.asarray([x[1] for x in results])

    return probs_batch, avg_value_batch

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        # self.Es = {}  # stores game.getGameEnded ended for board s
        # self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, state, env, obs, reward_discount_factor, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state (which should be a numpy array).

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # original_dict = copy.copy(self.Nsa)

        avg_value = 0
        for i in range(self.args.numMCTSSims):
            avg_value = i * avg_value + self.search(state, env, obs, False, reward_discount_factor)
            avg_value /= i + 1

        s = state.data.tobytes()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(env.action_space.n)]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, avg_value

        # if 0 in counts:
        #     logger.info("counts contains a zero: {}".format(counts))
        #     logger.info("current state: {}".format(s))
        #     logger.info("Nsa:\n{}".format(self.Nsa))
        #
        #     if str(s) in str(self.Nsa):
        #         logger.warning("the state IS in Nsa")
        #     else:
        #         logger.warning("the state IS NOT in Nsa")
        #
        #     added_values = {k: self.Nsa[k] for k in set(self.Nsa) - set(original_dict)}
        #     logger.warning("added values are:\n{}".format(added_values))

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs, avg_value

    def search(self, state, env, obs, done, reward_discount_factor):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        if done:
            # terminal node
            # logger.info("done")
            return 0

        s = state.data.tobytes()

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict_on_single_obs(obs)
            # valids = self.env.getValidMoves(state, 1)
            # self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                logger.warning("All valid moves were masked, do workaround.")
                # self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] = self.Ps[s] + env.action_space.n
                self.Ps[s] /= np.sum(self.Ps[s])

            # self.Vs[s] = valids
            self.Ns[s] = 0
            # return -v
            return v

        # valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(env.action_space.n):
            # if valids[a]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        # next_s, next_player = self.env.getNextState(state, 1, a)
        # next_s = self.env.getCanonicalForm(next_s, next_player)

        env.restore_full_state(state)
        next_state_obs, next_state_reward, next_state_done, info = env.step(a)
        next_state = env.clone_full_state()

        v = next_state_reward + reward_discount_factor * self.search(next_state, env, next_state_obs, next_state_done, reward_discount_factor)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        # return -v
        return v

