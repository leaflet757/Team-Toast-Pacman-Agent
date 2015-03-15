# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game, capture
import distanceCalculator
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  global teamIsRed
  teamIsRed = isRed
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    ''' 
    You should change this in your own agent.
    '''

    return random.choice(actions)

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    """
    CaptureAgent.registerInitialState(self, gameState)
    ''' 
    Your initialization code goes here, if you need any.
    '''
    # initializes opponents to track 
    # NOTE: 2 particle filters are being used for each agent
    self.opponents = self.getOpponents(gameState)
    self.filters = util.Counter()
    for opponent in self.opponents:
        self.filters[opponent] = createParticleFilter(opponent, gameState)
        global updateIndex
        if updateIndex == -1:
            updateIndex = self.index

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # Update Particle Filters
    if self.getPreviousObservation() != None:
        noisyDistances = gameState.getAgentDistances()
        prevNoisyDistances = self.getPreviousObservation().getAgentDistances()
        for opponent in self.opponents:
            #print noisyDistances[opponent] - prevNoisyDistances[opponent]
            if noisyDistances[opponent] - prevNoisyDistances[opponent] >= 13:
                self.filters[opponent].initializeUniformly(gameState)

    global updateIndex
    if updateIndex == self.index:
        self.updateFilters(gameState)
    

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def updateFilters(self, gameState):
    """
    Updates this agents particle filters for each corresponding 
    opponent agent.
    """
    # TODO:
    # if a ghost is kill, initialize filter uniformily
    '''
    if self.getPreviousObservation() != None and gameState.getScore() != self.getPreviousObservation().getScore():
        for opponent in self.opponents:
            self.filters[opponent].initializeUniformly(gameState)
    '''
    # perform particle filtering step
    # may need to be moved
    #if self.index == 1:
        #print gameState, 'score', gameState.getScore(), gameState.data.score, gameState.data.scoreChange

    # stall the game
    #for i in range(10000000):
        #pass

    positions = list()
    noisyDistances = gameState.getAgentDistances()
    self.debugClear()
    for opponent in self.opponents:
        #print 'self:', self.index, 'opponent: ', opponent, 'noise:', noisyDistances[opponent], 'position:', gameState.getAgentPosition(opponent)
        enemyPos = gameState.getAgentPosition(opponent)
        self.filters[opponent].observe(noisyDistances[opponent], gameState, self.distancer, self.index)
        self.filters[opponent].elapseTime(gameState, self.index)
        # check to see if we can observe the agent
        if enemyPos != None:
            #print 'test'
            self.filters[opponent].assignAgentPosition(enemyPos)
        for key, value in self.filters[opponent].getBeliefDistribution().items():
            if value > 0:
                #positions.append(key)
                #print key, value
                if opponent == 0:
                    positions.append(key)
                    self.debugDraw(key, [0.5,0,0])
                else:
                    self.debugDraw(key, [0,0.5,0])
        #positions.append(self.filters[opponent].getBeliefDistribution().argMax())
    # particle filtering debug
    #self.debugDraw(positions, [0.5,0,0], True);

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

##################
# InferenceModules
##################

class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """
  
  def __init__(self, agentIndex):
    "Sets the ghost agent for later access"
    self.index = agentIndex
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState. 
    You must first place the ghost in the gameState, using setGhostPosition below.

    NOTE:
    We do not know how the enemy agents will act so we say that the agent will move to any
    neighboring position with equal probability.
    """
    ghostPosition = gameState.getAgentPosition(self.index)
    actionDist = util.Counter()
    for a in gameState.getLegalActions( self.index ):
        actionDist[a] = 1.0
    actionDist.normalize()
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist

  def getPositionDist(self, gameState, ghostPosition):
    """
    Returns a distribution over successor positions of the ghost from the given gameState. 
    You must first place the ghost in the gameState, using setGhostPosition below.

    NOTE:
    We do not know how the enemy agents will act so we say that the agent will move to any
    neighboring position with equal probability.
    """
    walls = gameState.getWalls()
    neighbors = game.Actions.getLegalNeighbors(ghostPosition, walls)
    dist = util.Counter()
    for n in neighbors:
        dist[n] = 1.0
    dist.normalize()
    return dist

    """
    actionDist = util.Counter()
    for a in ['North', 'South', 'East', 'West', 'Stop']:
        actionDist[a] = 1.0
    actionDist.normalize()
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist
    """
  
  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index] = game.AgentState(conf, False)
    return gameState
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False)]
    self.initializeUniformly(gameState)
    
  # Methods that need to be overridden #
  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass
  
  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass
  
  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass
    
  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass


class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single ghost.
  
  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  """
  
  def initializeUniformly(self, gameState, numParticles=300):
    "Initializes a list of particles."
    self.numParticles = numParticles
    self.particles = util.Counter()
    for i in range(self.numParticles):
        position = random.choice(self.legalPositions)
        self.particles[position] += 1.0        
  
  def observe(self, observation, gameState, distancer, pacmanIndex):
    "Update beliefs based on the given distance observation."
    pacmanPosition = gameState.getAgentPosition(pacmanIndex)
    allPossible = util.Counter()
    for pos in self.legalPositions:
        trueDistance = util.manhattanDistance(pacmanPosition, pos) # NOTE: should different method be used?
        #trueDistance1 = distancer.getDistance(pacmanPosition, pos)
        #print 'true', trueDistance, trueDistance1
        allPossible[pos] = gameState.getDistanceProb(trueDistance, observation) * self.particles[pos]
    allPossible.normalize()
    self.particles = allPossible
    
  def elapseTime(self, gameState, pacmanIndex):
    """
    Update beliefs for a time step elapsing.
    
    As in the elapseTime method of ExactInference, you should use:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    to obtain the distribution over new positions for the ghost, given
    its previous position (oldPos) as well as Pacman's current
    position.
    """
    allPossible = util.Counter()
    for oldPos in self.legalPositions:
        newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
        #newPosDist = self.getPositionDist(gameState, oldPos)
        for newPos in newPosDist:
            allPossible[newPos] += newPosDist[newPos] * self.particles[oldPos]
    allPossible.normalize()
    self.particles = allPossible

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    return self.particles

  def assignAgentPosition(self, position):
    """
    Assign agents position with 100% certainty.
    """
    allPossible = util.Counter()
    allPossible[position] = 1.0
    self.particles = allPossible

  def reInitializeUniformly(self, gameState, numParticles=300):
    "Initializes a list of particles only on opponents side."
    """
    self.numParticles = numParticles
    self.particles = util.Counter()
    list = []
    global teamIsRed
    if teamIsRed:
        print 'yes'
        # igonore red side positions
    else:
        print 'no'
        list = capture.halfList(self.legalPositions, gameState.data.layout, not teamIsRed)
        print list
        # ignore blue side positions
    for i in range(self.numParticles):
        position = random.choice(list)
        self.particles[position] += 1.0
    """
    util.raiseNotDefined()


# Create a global particle filter that agents can share and update together
particleFilters = util.Counter()
updateIndex = -1 # used to only update on this agent
teamIsRed = False

def createParticleFilter(opponentIndex, gameState):
    """
    Creates a new particle filter if one does not already exist for 
    designated oppponentIndex.
    """
    if particleFilters[opponentIndex] == 0:
        particleFilters[opponentIndex] = ParticleFilter(opponentIndex)
        particleFilters[opponentIndex].initialize(gameState)
    return particleFilters[opponentIndex]