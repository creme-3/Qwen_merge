.PHONY: check compile project results plot

compile:
	python -m compileall -q eval merge scripts

project:
	python scripts/check_project.py

results:
	python scripts/check_results.py

plot:
	python scripts/plot_results.py

check: compile project results
