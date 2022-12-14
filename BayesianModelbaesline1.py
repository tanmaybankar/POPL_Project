from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
student_model = BayesianNetwork([('D', 'G'),
                                ('I', 'G'),
                                ('G', 'L'),
                                ('I', 'S')])
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
student_model.add_cpds(grade_cpd, difficulty_cpd,
                        intel_cpd, letter_cpd,
                        sat_cpd)


print(student_model.get_cpds()) #returns a list of Tabular CPDs
print(student_model.get_cardinality()) #variable cardinalities – If node is specified returns the cardinality of the node 
                                        #else returns a dictionary with the cardinality of each variable in the network

