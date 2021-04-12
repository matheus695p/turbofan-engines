![Build Status](https://www.repostatus.org/badges/latest/active.svg)

# turbofan-engines-predictive-mantenaince
end-to-end proyecto de mantenimiento predictivo


descripción:
  - baseline de modelo, definición de RUL como una parte constante más lineal
  - análisis de series de tiempo
 
  
### SETUP 

```sh
$ git clone https://github.com/matheus695p/turbofan-engines-predictive-mantenaince.git
$ cd turbofan-engines-predictive-mantenaince
$ echo instalar los requirements
$ pip install -r requirements.txt
```

```sh
│   README.md
│   readme.txt
│   requirements.txt
│
├───codes
│   ├───baseline
│   │       exploratory_baseline.py
│   │       __init__.py
│   │
│   ├───survival-analysis
│   │       survival.py
│   │       __init__.py
│   │
│   └───time-series
│           lagging_test.py
│           stationary_lagging.py
│           __init__.py
│
├───data
│       RUL_FD001.txt
│       RUL_FD002.txt
│       RUL_FD003.txt
│       RUL_FD004.txt
│       test_FD001.txt
│       test_FD002.txt
│       test_FD003.txt
│       test_FD004.txt
│       train_FD001.txt
│       train_FD002.txt
│       train_FD003.txt
│       train_FD004.txt
│
├───documents
│       Damage Propagation Modeling.pdf
│
└───src
    │   turbo_fan_module.py
    │   __init__.py
```
