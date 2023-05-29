PYTHON=python3
CLEANDATA=owid_sequences.pkl ar6_sequences.pkl
PRECIOUS=$(CLEANDATA) ar6_trajectories.pkl
ANALYSE=fig1-levels.png fig2-changes.png fig3_2D.png fig3_3D.png
CLASSIFY=single_variable_AUC.png single_variable_F1.png xbg-powerset.txt

.PRECIOUS: $(PRECIOUS)

all: fig1-levels.png xbg-powerset.txt

$(ANALYSE): figures.py data.py $(CLEANDATA)
	$(PYTHON) figures.py

$(CLASSIFY): xbg-powerset.py data.py $(CLEANDATA)
	$(PYTHON) xbg-powerset.py

owid_sequences.pkl: owid_sequences.py owid-co2-data.csv owid_notcountry.csv
	$(PYTHON) owid_sequences.py
	
ar6_sequences.pkl: ar6_sequences.py ar6_trajectories.pkl
	$(PYTHON) owid_sequences.py

ar6_trajectories.pkl: ar6_trajectories.py AR6_Scenarios_Database_ISO3_v1.1.csv
	$(PYTHON) ar6_trajectories.py

clean:
	-rm -f $(ANALYSE) $(CLASSIFY)

cleaner: clean
	-rm -f $(PRECIOUS)
