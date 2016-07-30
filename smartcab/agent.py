import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

from math import log,e
from numpy import mean
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Add the states that are required
        lightStates = ["red", "green"]
        actionStates = [None, "forward", "left", "right"]
        recommendingStates = ["forward", "left", "right"]
    
        # The Q-table we want to create will be dictionary as per the python's datatype
        # So, we need to create the keys and values for that dictionary QTable
        QKeys = []
        # Key structure (recommendedState, lightState, oncomingState, leftState, actionState)
        for actionState in actionStates:
            for leftState in actionStates:
                for oncomingState in actionStates:
                    for lightState in lightStates:
                        for recommendedState in recommendingStates:
                            QKeys.append((recommendedState, lightState, oncomingState, leftState,actionState))
        QValues = [random.random() * 4 for x in xrange(len(QKeys))]
        
        self.QTable = dict(zip(QKeys, QValues))
        self.totalReward = 0.0
        self.totalRewards = []
        self.avgRewards = []
        self.explorationRateFunc = []
        self.explorationRate = e/5
        self.remainingTime = 0
        self.count = 0
        self.successCount = 0
        self.trialCount = 0
        self.numActions = 1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print "Total Reward = ",self.totalReward/self.numActions
        
        # If the agent reaches before deadline and the reward is positive, update the exploration rate
        if self.remainingTime > 0 and self.totalReward > 0: self.count += 1
        if self.remainingTime > 0 : self.successCount += 1
        self.explorationRate = e**-((3+self.count)/5.0)

        self.trialCount += 1
        self.avgRewards.append(mean(self.totalRewards))
        self.totalRewards.append(self.totalReward/self.numActions)
        self.explorationRateFunc.append(self.explorationRate)
        self.totalReward = 0.0
        self.numActions = 1
        
        if self.trialCount == 100:
            print "FINAL STATS:"
            print "SUCCESS COUNT:",self.successCount+1,"/ 100"
            plt.figure(1)
            plt.subplot(311)
            plt.plot(self.totalRewards)
            plt.title('Learning Performance')
            plt.xlabel('Trials')
            plt.ylabel('Total Rewards')
            plt.subplot(312)
            plt.plot(self.avgRewards)
            plt.xlabel('Trials')
            plt.ylabel('Average Rewards')
            plt.subplot(313)
            plt.plot(self.explorationRateFunc)
            plt.xlabel('Trials')
            plt.ylabel('Exploration Rate')
            plt.show()
            print self.QTable

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        
        # TODO: Select action according to your policy
        decisionTable = {None: self.QTable[(self.state + (None,))], 'right': self.QTable[(self.state + ('right',))], 'left': self.QTable[(self.state + ('left',))], 'forward': self.QTable[(self.state + ('forward',))]}
        
        
        if  self.explorationRate < random.random():
            qValPresent, action = max((v, k) for k, v in decisionTable.iteritems())
        else:
            action = random.choice([None, 'forward', 'left', 'right'])
            qValPresent = decisionTable [(action)]
        
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        learningRate = 0.75
        discountRate = 0.4
        
        newInputs = self.env.sense(self)
        newState = (self.next_waypoint, newInputs['light'], newInputs['oncoming'], newInputs['left'])
        
        futureTable = {None: self.QTable[(newState + (None,))], 'right': self.QTable[(newState + ('right',))], 'left': self.QTable[(newState + ('left',))], 'forward': self.QTable[(newState + ('forward',))]}
        
        futureReward, futureAction = max((v, k) for k, v in futureTable.iteritems())
        improvedEstimate = reward + discountRate * futureReward
        
        self.QTable[(self.state + (action,))] = qValPresent + learningRate * (improvedEstimate - qValPresent)

        self.totalReward += reward
        self.numActions += 1
        self.remainingTime = deadline
        

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line



if __name__ == '__main__':
    run()

