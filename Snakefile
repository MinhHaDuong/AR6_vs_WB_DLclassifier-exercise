# Snakefile for the classifying country scenarios project
#
# created 2023-06-01 by haduong@centre-cired.fr


# Data wrangling

rule ar6_trajectories:
    input: "AR6_Scenarios_Database_ISO3_v1.1.csv"
    output: "ar6_trajectories.pkl"
    shell: "python3 ar6_trajectories.py"

rule ar6_sequences:
    input: "ar6_trajectories.pkl"
    output: "ar6_sequences.pkl"
    shell: "python3 ar6_sequences.py"

rule owid_sequences:
    input: "owid-co2-data.csv", "owid_notcountry.csv"
    output: "owid_sequences.pkl"
    shell: "python3 owid_sequences.py"


# Analysis

SEQUENCES=["owid_sequences.pkl", "ar6_sequences.pkl"]
SCRIPTS = ['powerset', 'compare_classifiers', 'compare_reductions']

rule run_script:
    input: SEQUENCES
    output: "{name}.pkl"
    shell: "python3 {wildcards.name}.py"
    wildcard_constraints: name = '|'.join(SCRIPTS)


# Visualisation

FIGURES_UNIVAR=["figures/" + name + ".png" for name in ["fig1-levels", "fig2-changes", "fig3_2D", "fig3_3D", "fig4_cdf"]]
FIGURES_SENSITIVITY=["figures/" + name + ".png" for name in ["single_variable_AUC", "single_variable_F1"]]
TABLES=["tables/" + name + ".csv" for name in ["powerset", "compare_classifiers", "compare_reductions"]]

rule create_table:
    input: "{name}.pkl"
    output: "tables/{name}.csv"
    shell: "python3 {wildcards.name}_tables.py"
    wildcard_constraints: name = '|'.join(SCRIPTS)

rule figures_univar:
    input: SEQUENCES
    output: FIGURES_UNIVAR
    script: "data_figures.py"

rule figures_sensitivity:
    input: SEQUENCES
    output: FIGURES_SENSITIVITY
    script: "powerset_figures.py"


# Housecleaning

rule all:
    input: FIGURES_UNIVAR, TABLES, FIGURES_SENSITIVITY

rule clean:
    shell: "rm -f " + " ".join(FIGURES_UNIVAR + FIGURES_SENSITIVITY + TABLES)
