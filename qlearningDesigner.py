import random,util,math,copy
from unitSpaceCFD import unitSpaceAgent

class QLearningDesigner(unitSpaceAgent):
    def __init__(self, alpha = 0.5, initialEpsilon = 0.5, gamma = 0.8):
        #unitSpace.__init__(self)
        self.qvalue = util.Counter()
        self.alpha = alpha
        self.epsilon = initialEpsilon
        self.discount = gamma

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if not (state,action) in self.qvalue: # haven't visited this (state,action) before, so register it as 0.0
            self.qvalue[(state, action)] = 0.0
        return self.qvalue[(state, action)]

    def computeValueFromQValues(self, state, currentLegalActions):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        if not len(currentLegalActions):
            print("No Legal Action!!")
            return 0.0
        else:
            return max([self.getQValue(state,action) for action in currentLegalActions])

    def computeActionFromQValues(self, state, currentLegalActions):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        policy = util.Counter()
        for action in currentLegalActions:
            policy[action] = self.getQValue(state, action)
        return policy.argMax()  # pi(s) = argmax(Q(s,a))

    def getAction(self, state, location, occupiedCell, occupiedBoundary):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state,location,occupiedCell,occupiedBoundary)
        action = None
        if len(legalActions) == 0:
            return action
        elif util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state, legalActions)
        return action

    def update(self, state, location, action, nextState, occupiedCell, occupiedBoundary, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        nextLocation = self.newLocationAfterAction(location, action)
        nextStatelegalActions = self.getLegalActions(nextState, nextLocation, occupiedCell, occupiedBoundary)
        #updating function for qvalue:
        self.qvalue[(state, action)] = \
            self.qvalue[(state, action)] + self.alpha*(reward + self.discount*self.getValue(nextState, nextStatelegalActions)-self.qvalue[(state, action)])

    def getPolicy(self, state, curLegalActions):
        return self.computeActionFromQValues(state, curLegalActions)

    def getValue(self, state, curLegalActions):
        return self.computeValueFromQValues(state, curLegalActions)

