# Snakefile for the classifying country scenarios project
#
# created 2023-06-01 by haduong@centre-cired.fr


# Directory structure
data_dir = "data"
cache_dir = f"cache"
figures_dir = "figures"
tables_dir = f"tables"


# Data wrangling
rule ar6_trajectories:
    input: f"{data_dir}/AR6_Scenarios_Database_ISO3_v1.1.csv"
    output: f"{cache_dir}/ar6_trajectories.pkl"
    shell: "python3 ar6_trajectories.py"

rule ar6_sequences:
    input: f"{cache_dir}/ar6_trajectories.pkl"
    output: f"{cache_dir}/ar6_sequences.pkl"
    shell: "python3 ar6_sequences.py"

rule owid_sequences:
    input:
        f"{data_dir}/owid-co2-data.csv",
        f"{data_dir}/owid_notcountry.csv"
    output: f"{cache_dir}/owid_sequences.pkl"
    shell: "python3 owid_sequences.py"


# Analysis
SEQUENCES = [f"{cache_dir}/owid_sequences.pkl", f"{cache_dir}/ar6_sequences.pkl"]
SCRIPTS = ['powerset', 'compare_classifiers', 'compare_reductions']

wildcard_constraints:
    script= '|'.join(SCRIPTS)

rule run_script:
    input: SEQUENCES
    output: f"{cache_dir}/{{script}}.pkl"
    shell: "python3 {{script}}.py"


# Visualisation
FIGURES_UNIVAR = [f"{figures_dir}/{name}.png" for name in ["fig1-levels", "fig2-changes", "fig3_2D", "fig3_3D", "fig4_cdf"]]
FIGURES_SENSITIVITY = [f"{figures_dir}/{name}.png" for name in ["single_variable_AUC", "single_variable_F1"]]

POWERSET_TABLE = f"{tables_dir}/powerset.tex"
CLASSIFIER_TABLES = [f"{tables_dir}/compare_classifiers_{variant}.tex" for variant in [
    "raw", "parallel_raw",
    "base", "parallel_base",
    "balanced", "parallel_balanced",
    "normalized", "parallel_normalized"]
]
REDUCTION_TABLES = [f"{tables_dir}/compare_reductions_{variant}.tex" for variant in [
    "normalized", "PCA", "latent", "latent2"]
]
TABLES = CLASSIFIER_TABLES + REDUCTION_TABLES + [POWERSET_TABLE]

rule create_powerset_table:
    input: f"{cache_dir}/powerset.pkl"
    output: POWERSET_TABLE
    shell: "python3 powerset_tables.py"

rule create_classifier_tables:
    input: f"{cache_dir}/compare_classifiers.pkl"
    output: CLASSIFIER_TABLES
    shell: "python3 compare_classifiers_tables.py"

rule create_reduction_tables:
    input: f"{cache_dir}/compare_reductions.pkl"
    output: REDUCTION_TABLES
    shell: "python3 compare_reductions_tables.py"

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
