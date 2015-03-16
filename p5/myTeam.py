# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Agent
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
    # self.debugClear()
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
                    #self.debugDraw(key, [0.5,0,0])
                else:
                    pass
                    #self.debugDraw(key, [0,0.5,0])
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

####################
# InferenceModules #
####################

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

##################
# Search Modules #
##################

class SearchAgent(Agent):
  """
  This very general search agent finds a path using a supplied search algorithm for a
  supplied search problem, then returns actions to follow that path.
  
  As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)
  
  Options for fn include:
    depthFirstSearch or dfs
    breadthFirstSearch or bfs
    
  
  Note: You should NOT change any code in SearchAgent
  """
    
  def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
    # Warning: some advanced Python magic is employed below to find the right functions and problems
    
    # Get the search function from the name and heuristic
    if fn not in dir(search): 
      raise AttributeError, fn + ' is not a search function in search.py.'
    func = getattr(search, fn)
    if 'heuristic' not in func.func_code.co_varnames:
      print('[SearchAgent] using function ' + fn) 
      self.searchFunction = func
    else:
      if heuristic in dir(searchAgents):
        heur = getattr(searchAgents, heuristic)
      elif heuristic in dir(search):
        heur = getattr(search, heuristic)
      else:
        raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
      print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic)) 
      # Note: this bit of Python trickery combines the search algorithm and the heuristic
      self.searchFunction = lambda x: func(x, heuristic=heur)
      
    # Get the search problem type from the name
    if prob not in dir(searchAgents) or not prob.endswith('Problem'): 
      raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
    self.searchType = getattr(searchAgents, prob)
    print('[SearchAgent] using problem type ' + prob) 
    
  def registerInitialState(self, state):
    """
    This is the first time that the agent sees the layout of the game board. Here, we
    choose a path to the goal.  In this phase, the agent should compute the path to the
    goal and store it in a local variable.  All of the work is done in this method!
    
    state: a GameState object (pacman.py)
    """
    if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
    starttime = time.time()
    problem = self.searchType(state) # Makes a new search problem
    self.actions  = self.searchFunction(problem) # Find a path
    totalCost = problem.getCostOfActions(self.actions)
    print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
    if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)
    
  def getAction(self, state):
    """
    Returns the next action in the path chosen earlier (in registerInitialState).  Return
    Directions.STOP if there is no further action to take.
    
    state: a GameState object (pacman.py)
    """
    if 'actionIndex' not in dir(self): self.actionIndex = 0
    i = self.actionIndex
    self.actionIndex += 1
    if i < len(self.actions):
      return self.actions[i]    
    else:
      return Directions.STOP

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()         

class PositionSearchProblem(SearchProblem):
  """
  A search problem defines the state space, start state, goal test,
  successor function and cost function.  This search problem can be 
  used to find paths to a particular point on the pacman board.
  
  The state space consists of (x,y) positions in a pacman game.
  
  Note: this search problem is fully specified; you should NOT change it.
  """
  
  def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
    """
    Stores the start and goal.  
    
    gameState: A GameState object (pacman.py)
    costFn: A function from a search state (tuple) to a non-negative number
    goal: A position in the gameState
    """
    self.walls = gameState.getWalls()
    self.startState = gameState.getPacmanPosition()
    if start != None: self.startState = start
    self.goal = goal
    self.costFn = costFn
    if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
      print 'Warning: this does not look like a regular search maze'

    # For display purposes
    self._visited, self._visitedlist, self._expanded = {}, [], 0

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
     isGoal = state == self.goal 
     
     # For display purposes only
     if isGoal:
       self._visitedlist.append(state)
       import __main__
       if '_display' in dir(__main__):
         if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
           __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
       
     return isGoal   
   
  def getSuccessors(self, state):
    """
    Returns successor states, the actions they require, and a cost of 1.
    
     As noted in search.py:
         For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextState = (nextx, nexty)
        cost = self.costFn(nextState)
        successors.append( ( nextState, action, cost) )
        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if state not in self._visited:
      self._visited[state] = True
      self._visitedlist.append(state)
      
    return successors

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    if actions == None: return 999999
    x,y= self.getStartState()
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]: return 999999
      cost += self.costFn((x,y))
    return cost

def manhattanHeuristic(position, problem, info={}):
  "The Manhattan distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
  "The Euclidean distance heuristic for a PositionSearchProblem"
  xy1 = position
  xy2 = problem.goal
  return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

class FoodSearchProblem:
  """
  A search problem associated with finding the a path that collects all of the 
  food (dots) in a Pacman game.
  
  A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
    pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
    foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food 
  """
  def __init__(self, startingGameState):
    self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
    self.walls = startingGameState.getWalls()
    self.startingGameState = startingGameState
    self._expanded = 0
    self.heuristicInfo = {} # A dictionary for the heuristic to store information
      
  def getStartState(self):
    return self.start
  
  def isGoalState(self, state):
    return state[1].count() == 0

  def getSuccessors(self, state):
    "Returns successor states, the actions they require, and a cost of 1."
    successors = []
    self._expanded += 1
    for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
      x,y = state[0]
      dx, dy = Actions.directionToVector(direction)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.walls[nextx][nexty]:
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
    return successors

  def getCostOfActions(self, actions):
    """Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999"""
    x,y= self.getStartState()[0]
    cost = 0
    for action in actions:
      # figure out the next state and see whether it's legal
      dx, dy = Actions.directionToVector(action)
      x, y = int(x + dx), int(y + dy)
      if self.walls[x][y]:
        return 999999
      cost += 1
    return cost

class AStarFoodSearchAgent(SearchAgent):
  "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
  def __init__(self):
    self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
    self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
  """
  Your heuristic for the FoodSearchProblem goes here.
  
  This heuristic must be consistent to ensure correctness.  First, try to come up
  with an admissible heuristic; almost all admissible heuristics will be consistent
  as well.
  
  If using A* ever finds a solution that is worse uniform cost search finds,
  your heuristic is *not* consistent, and probably not admissible!  On the other hand,
  inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
  
  The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a 
  Grid (see game.py) of either True or False. You can call foodGrid.asList()
  to get a list of food coordinates instead.
  
  If you want access to info like walls, capsules, etc., you can query the problem.
  For example, problem.walls gives you a Grid of where the walls are.
  
  If you want to *store* information to be reused in other calls to the heuristic,
  there is a dictionary called problem.heuristicInfo that you can use. For example,
  if you only want to count the walls once and store that value, try:
    problem.heuristicInfo['wallCount'] = problem.walls.count()
  Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
  """
  position, foodGrid = state
  "*** YOUR CODE HERE ***"
  from util import manhattanDistance
  from search import aStarSearch
  numFood = 0
  distances = []
  distanceMin = 0
  distanceMax = 0
  foodPos = {}
  foodPos[0] = position
  for food in foodGrid.asList():
      numFood += 1
      distance = manhattanDistance(position, food)
      distances.append(distance)
      foodPos[distance] = food
  if distances:
      distanceMin = min(distances)
      distanceMax = max(distances)
  foodMin = foodPos[distanceMin]
  foodMax = foodPos[distanceMax]
  #determine best possible path length from 'closest' food to 'farthest' food
  distanceMinMax = manhattanDistance(foodPos[distanceMin], foodPos[distanceMax])
  #return above path length plus best possible path length to closest food
  return (distanceMin + distanceMinMax)
  
def aStarSearch(problem, heuristic=foodHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  from util import PriorityQueue
  from game import Directions
  state = problem.getStartState()
  successor = (state, Directions.STOP, 0)
  prev = successor
  statePQueue = PriorityQueue()
  statePQueue.push(successor, 0)
  goalPath = []
  successorActionParent = {}
  visitedStates = []
  while not statePQueue.isEmpty(): #While there are states to expand
      successor = statePQueue.pop()
      if problem.isGoalState(successor[0]):  #Are we at the goal?
          curr = successor[0]
          while problem.getStartState() != curr: #Get actions that we took to get to goal
              actionParent = successorActionParent[curr]
              goalPath.append(actionParent[0])
              curr = actionParent[1]
              curr = curr[0]
          goalPath.reverse()
          return goalPath
      if successor[0] not in visitedStates: #Cycle prevention
          visitedStates.append(successor[0])
          successors = problem.getSuccessors(successor[0])
          for x in successors: #add states to PQueue of states to expand
              if x[0] not in successorActionParent:
                  successorActionParent[x[0]] = (x[1], successor)
              cost = 0
              costPath = []
              curr = x[0]
              while problem.getStartState() != curr: #Get actions that we took to get here
                  actionParent = successorActionParent[curr]
                  costPath.append(actionParent[0])
                  curr = actionParent[1]
                  curr = curr[0]
              costPath.reverse()
              cost = problem.getCostOfActions(costPath) #Get total cost of actions to get here
              cost = cost + heuristic(x[0], problem) #Add heuristic to cost
              statePQueue.push(x, cost)
  return [Directions.STOP]