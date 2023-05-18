# Are IPCC AR6 scenarios realistic?

Author: haduong@centre-cired.fr

2023-05

We define a trajectory as a matrix with 6 columns and up to 4 rows.
The rows correspond to variables: CO2 emissions, GDP, populations, primary energy
The columns correspond to years, with a 5 years difference so that the trajectory is 25 years

The simulation trajectories are picked from the IPCC AR6 national scenario database
The observations trajectories are picked from owid-co2 dataset

We pool all the model, regions, years into two big sets of trajectories,
and trained five kind of machine learning binary classifiers
to distinguish simulations from observations.

Examining the figures produced by data.py show that:
- There are 'World' regions in the datasets.
- There are problems with a few 'GDP|MER' simulation series.
- There are problems with some 'Population' simulation series.

Results of the various model_*.py shows that:
- The GBM classifier works best. 

Results of the xbg-powerset.py show that:
- Simulations are very distinguishable from observations.
- Trajectories with the 'population' variable are more distinguishable that those without

Ideas:

Pour des ensembles de variables
Chercher les variables distinguables vs. indistinguables (e.g. realistically modeled)
en individuelle, en paire, en trio, ensemble.
Quelles méthodes vérifient les critères
P1: Dans un ensemble de variables indistinguables, tous les sous ensembles le sont 
Corolaire: L'intersection de deux ensembles indistinguables l'est aussi
P2: L'union de deux ensembles indistinguables ne l'est pas nécessairement
Corolaire: il peut y avoir plusieurs ensembles de variables indistinguables maximaux
P3: Tout ensemble contenant au moins une variable distinguable est distinguable 
Corolaire: ce type de structure a certainement un nom mathématique...

H1: Il existe des variables indistinguables dans la base de scénarios AR6
H2: Les variables normatives sont distinguables
Corolaire:
- Liste des ensembles de variables indistinguables maximaux
- Leur intersection: le coeur économique

Labeliser Baseline/Constrained et le SSP
-> H0: Les scénarios de baseline sont plus proches du passé que les scénarios d'action
-> H1: Les cinq SSP sont aussi proches de la réalité les uns des autres
-> H2: Plus l'action est forte plus les scénarios s'éloignent de la réalité

