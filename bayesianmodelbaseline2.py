

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

student_model = BayesianNetwork([('D', 'G'),
                                ('I', 'G'),
                                ('G', 'L'),
                                ('I', 'S')])
import networkx as nx
import pylab as plt
nx_graph = nx.DiGraph(student_model.edges())
nx.draw(nx_graph, with_labels=True)
plt.show()

grade_cpd = TabularCPD(
                        variable='G',
                        variable_card=3,
                        values=[[0.3, 0.05, 0.9, 0.5],
                        [0.4, 0.25, 0.08, 0.3],
                        [0.3, 0.7, 0.02, 0.2]],
                        evidence=['I', 'D'],
                        evidence_card=[2, 2])
difficulty_cpd = TabularCPD(
                            variable='D',
                            variable_card=2,
                            values=[[0.6], [0.4]])
intel_cpd = TabularCPD(
                        variable='I',
                        variable_card=2,
                        values=[[0.7], [0.3]])
letter_cpd = TabularCPD(
                        variable='L',
                        variable_card=2,
                        values=[[0.1, 0.4, 0.99],
                        [0.9, 0.6, 0.01]],
                        evidence=['G'],
                        evidence_card=[3])
sat_cpd = TabularCPD(
                    variable='S',
                    variable_card=2,
                    values=[[0.95, 0.2],
                    [0.05, 0.8]],
                    evidence=['I'],
                    evidence_card=[2])


#adding cpds
student_model.add_cpds(grade_cpd, difficulty_cpd,
                        intel_cpd, letter_cpd,
                        sat_cpd)

# We can now call some methods on the BayesianNetwork object.
print(student_model.get_cpds('G'))#returns a list of Tabular CPDs
# print(student_model.get_cardinality())   #variable cardinalities – If node is specified returns the cardinality of the node 
                                           #else returns a dictionary with the cardinality of each variable in the network

# results with Variable Elimination Algorithm




infer = VariableElimination(student_model)


result = infer.query(['G'], evidence={'I': 0, 'S': 1})
print(result)

###Monty Hall problem

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Defining the network structure
monty = BayesianNetwork([("C", "H"), ("P", "H")])

# Defining the CPDs:
cpd_c = TabularCPD("C", 3, [[0.33], [0.33], [0.33]])
cpd_p = TabularCPD("P", 3, [[0.33], [0.33], [0.33]])
cpd_h = TabularCPD(
    "H",
    3,
    [
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
    ],
    evidence=["C", "P"],
    evidence_card=[3, 3],
)

# Associating the CPDs with the network structure.
monty.add_cpds(cpd_c, cpd_p, cpd_h)

# Some other methods
print(monty.get_cpds())

nx_graph = nx.DiGraph(monty.edges())
nx.draw(nx_graph, with_labels=True)
plt.show()

monty.check_model()

# Infering the posterior probability

infer = VariableElimination(monty)
posterior_p = infer.query(["P"], evidence={"C": 0, "H": 2})
print(posterior_p)