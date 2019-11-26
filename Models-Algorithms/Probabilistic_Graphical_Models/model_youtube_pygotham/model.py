# # https://www.youtube.com/watch?v=DEHqIxX1Kq4&feature=youtu.be

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

olympic_model = BayesianModel(
    [
        ('Genetics', 'OlympicTrials'),
        ('Practice', 'OlympicTrials'),
        ('OlympicTrials', 'Offer'),
    ]
)

genetics_cpd = TabularCPD(
    variable='Genetics',
    variable_card=2, # Number of possibilities of the variable
    values=[[.2, .8]]
)

practice_cpd = TabularCPD(
    variable='Practice',
    variable_card=2,
    values=[[.7, .3]]
)

olympic_trials_cpd = TabularCPD(
    variable='OlympicTrials',
    variable_card=3,
    values=[
        [.5, .8, .8, .9],
        [.3, .15, .1, .08],
        [.2, .05, .1, .02],
    ],
    evidence=['Genetics', 'Practice'], # Parents
    evidence_card=[2,2]
)

offer_cpd = TabularCPD(
    variable='Offer',
    variable_card=2,
    values=[
        [.95, .8, .5],
        [.05, .2, .5]
    ],
    evidence=['OlympicTrials'],
    evidence_card=[3]
)

olympic_model.add_cpds(genetics_cpd, practice_cpd, offer_cpd, olympic_trials_cpd)

# Examine structure of Graph
print(olympic_model.get_cpds())
