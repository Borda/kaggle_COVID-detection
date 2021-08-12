# Kaggle: [COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection)

[![CI complete testing](https://github.com/Borda/kaggle_COVID-detection/actions/workflows/ci_testing.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_COVID-detection/actions/workflows/ci_testing.yml)
[![Code formatting](https://github.com/Borda/kaggle_COVID-detection/actions/workflows/code-format.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_COVID-detection/actions/workflows/code-format.yml)
[![codecov](https://codecov.io/gh/Borda/kaggle_COVID-detection/branch/main/graph/badge.svg)](https://codecov.io/gh/Borda/kaggle_COVID-detection)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/kaggle_COVID-detection/main.svg)](https://results.pre-commit.ci/latest/github/Borda/kaggle_COVID-detection/main)

In this competition, you’ll identify and localize COVID-19 abnormalities on chest radiographs.
In particular, you'll categorize the radiographs as negative for pneumonia or typical, indeterminate, or atypical for COVID-19.
Organizers provided dataset - imaging data and annotations from a group of radiologists.

![Sample images](./assets/image-class-samples.jpg)

In other words the task to solve is image classification accompanied by attention

- user shall classify patient COVID19 situation and also tell why think so, what are the regions in the scan that makes him think this case is positive.

![Label distribution](./assets/labels-pie.png)

## Experimentation

### install this tooling

A simple way how to use this basic functions:

```bash
! pip install https://github.com/Borda/kaggle_COVID-detection/archive/main.zip
```

### run notebooks in Kaggle

- [COVID199 detection with Flash ⚡](https://www.kaggle.com/jirkaborovec/covid-detection-with-lightning-flash)
- [COVID199 detection - predictions](https://www.kaggle.com/jirkaborovec/covid-detection-with-lightning-flash-predictions)

### some results

Training progress with ResNet50 with training  for 50 epochs:

![Training process](./assets/logging-loss.png)
![Training process](./assets/logging-metric.png)
