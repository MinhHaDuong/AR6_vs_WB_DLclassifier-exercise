PYTHON=python3
CLEANDATA=owid_sequences.pkl ar6_sequences.pkl
PRECIOUS=$(CLEANDATA) ar6_trajectories.pkl

FIGURES_UNIVAR=fig1-levels.png fig2-changes.png fig3_2D.png fig3_3D.png fig4_cdf.png
FIGURE_SENSITIVITY=single_variable_AUC.png single_variable_F1.png
TABLES=xbg-powerset.txt classifiers_compare_kind.txt classifiers_compare_autoencoder.txt

.PRECIOUS: $(PRECIOUS)

all: $(FIGURES_UNIVAR) $(TABLES) $(FIGURE_SENSITIVITY)

$(FIGURES_UNIVAR): figures.py data.py $(CLEANDATA)
	$(PYTHON) figures.py

$(FIGURE_SENSITIVITY) xbg-powerset.txt: xbg-powerset.py data.py $(CLEANDATA)
	$(PYTHON) xbg-powerset.py

classifiers_compare_kind.txt classifiers_compare_kind.pkl: classifiers_compare_kind.py data.py $(CLEANDATA)
	$(PYTHON) classifiers_compare_kind.py

classifiers_compare_autoencoder.txt: classifiers_compare_autoencoder.py data.py $(CLEANDATA)
	$(PYTHON) classifiers_compare_autoencoder.py

owid_sequences.pkl: owid_sequences.py owid-co2-data.csv owid_notcountry.csv
	$(PYTHON) owid_sequences.py

ar6_sequences.pkl: ar6_sequences.py ar6_trajectories.pkl
	$(PYTHON) owid_sequences.py

ar6_trajectories.pkl: ar6_trajectories.py AR6_Scenarios_Database_ISO3_v1.1.csv
	$(PYTHON) ar6_trajectories.py

clean:
	-rm -f $(FIGURES_UNIVAR) $(FIGURE_SENSITIVITY) $(TABLES) classifiers_compare_kind.pkl

cleaner: clean
	-rm -f $(PRECIOUS)
