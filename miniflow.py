import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add tis Node as an outbound Node to _that_ Node
        for node in inbound_nodes:
            node.outbound_nodes.append(self)
        # The output value
        self.value = None

    def forward(self):
        '''
        Forward propagation.

        Compute the output value based on `inbound_nodes` and store the result in self.value.
        '''
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes, so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value may be passed as an argument to forward().
    # All other node implementations should get the value of the previous node from self.inbound_nodes
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        # NOTE: The weights and bias properties here are not numbers, but rather references to other nodes.
        # The weight and bias values are stored within the respective nodes.
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        # Apply linear transform using numpy
        self.value = np.add(np.matmul(inputs, weights), bias)

class Sigmoid(Node):
    def __init__(self, x):
        Node.__init__(self, [x])

    def _sigmoid(self, x):
        '''
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.
        '''
        return 1 / (1 + np.exp(-x))

    def forward(self):
        '''
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        '''
        self.value = self._sigmoid(self.inbound_nodes[0].value)

def topological_sort(feed_dict):
    '''
    Sort generic nodes in topological order using Kahn's Algorithm (https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm).

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    Exemple:
        x = Input()
        y = Input()
        feed_dict = {x: 5, y: 10}

    Returns a list of sorted nodes.
    '''

    input_nodes = [n for n in feed_dict.keys()]

    # 1. Build a temporary set of nodes to which Kahn's algo will be applied
    G = {}
    graph = [n for n in input_nodes]
    while len(graph) > 0:
        n = graph.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[m]['in'].add(n)
            G[n]['out'].add(m)
            graph.append(m)

    # 2. Kanh's algorithm
    # Set of all nodes with no incoming edge
    S = set(input_nodes)
    # Empty list that will contain the sorted elements
    L = []
    while len(S) > 0:
        # Remove a node n from S
        n = S.pop()
        # Retrieve value for Input nodes
        if isinstance(n, Input):
            n.value = feed_dict[n]
        # Add n to tail of L
        L.append(n)
        for m in n.outbound_nodes:
            # Remove edge e (n --> m) from the graph
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # If m has no other incoming edges
            if (len(G[m]['in']) == 0):
                S.add(m)

    # 3. Return the topological sorted list of Nodes
    return L

def forward_pass(output_node, sorted_nodes):
    for n in sorted_nodes:
        n.forward()

    return output_node.value
