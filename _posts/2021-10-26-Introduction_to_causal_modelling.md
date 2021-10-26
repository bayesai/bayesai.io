---
layout: post
title:  "Introducing Causal Modelling"
date:   2021-10-26 07:00:14 +0000
excerpt_separator: <!--more-->
published: true
mathjax: true
categories:
  - "graphical_Models"
tags:
  - causal-models, bayesian networks, pgmpy, i-maps
author:
  - Pavan
---

# Introduction to causal modelling part 1


```python
from IPython.display import Image
import pandas as pd
import numpy as np
import random
```


## Data generation code


```python
people = ['Rheia','Kronos','Demeter',
          'Hades','Hestia','Poseidon',
          'Hera','Zeus','Artemis',
          'Apollo','Leto','Ares',
          'Athena','Hephaestus','Aphrodite',
          'Cyclope','Persephone','Hermes','Hebe','Dionysus']

y0 = [0,1,0,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,1,1]
y1 = [1,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0]
data = pd.DataFrame({'people':people,'y^a=0':y0,'y^a=1':y1})
```

## Section 1: Introducing some terminology

* We start with a simple example where there are two types of treatment denoted $$a=0$$ and $$a=1$$.
* For each individual in our population we let $$Y^{a=0}$$ and $$Y^{a=1}$$ denote the outcomes of each treatment, which we assume is deterministic for each individual. 
* These variables are called counterfactual because at any moment in time for an individual we can only observe one of $$Y^{a=i}$$ by assigning that particular treatment. 

### Section 1.1: Causal Effects



$$\textbf{Definition 1.1.1}$$: We say that there is an individual causal effect if $$Y^{a=0} \neq Y^{a=1}$$

* As mentioned above individual causal effects cannot be identified because only one potential outcome can be observed.

* We pretend as though we are in a world where we can observe both $$Y^{a=1}$$ and $$Y^{a=0}$$ for purposes of illustration.

* The sample data is shown below.

* For each individual and treatment $$j=0,1$$ the variable $$Y^{a=j}=1$$ if the outcome is a success under this treatment else $$0$$. We do not attach a particular application to the data to keep the interpretations flexible. 




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: left;
    }
</style>
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Row</th>
      <th>Person</th>
      <th>$$y^a=0$$</th>
      <th>$$y^a=1$$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rheia</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Kronos</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Demeter</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hades</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hestia</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Poseidon</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hera</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Zeus</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Artemis</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Apollo</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Leto</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ares</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Athena</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Hephaestus</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Aphrodite</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Cyclope</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Persephone</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hermes</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Hebe</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Dionysus</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* Since we cannot observe individual causal effects we focus on population causal effects introduced in the next section. 

* Note above in real life we would NOT have access to everyones full data but to introduce the concepts in the next few sections its useful to pretend that we do. 

## Section 2: Average Causal effects

* We said that by assumption $$Y^{a=1}$$ at individual level is NOT random. However consider choosing an individual from the population uniformaly at random. We can talk about $$P(Y^{a=i}=j)$$ which is the probability that a random person from the population has outcome $$j$$ under treatment $$i$$.  

* Let us calculate the probabilities $$P(Y^{a=1} = 1)$$ and $$P(Y^{a=0}=1)$$ from our data, since in this hypothetical world we know the outcomes for each individual.




```python
PY_a1 = data['y^a=1'].sum()/data.shape[0]
PY_a0 = data['y^a=0'].sum()/data.shape[0]

print('P(Y^(a=0)=1)='+str(PY_a0))
print('P(Y^(a=1)=1)='+str(PY_a1))
```

$$P(Y^{a=0} = 1) = 0.5$$

$$P(Y^{a=1} =1) = 0.5$$


$$\textbf{Definition 2.1}$$: $$\textbf{Causal Null hypothesis}$$: No average causal effect i.e. $$P(Y^{a=1}=1) = P(Y^{a=1}=0)$$. 

* So there is a causal effect if $$P(Y^{a=1}=1) \neq P(Y^{a=0}=1)$$. Or equivalently for binary random variables if $$\mathbb{E}(Y^{a=1}) \neq \mathbb{E}(Y^{a=0})$$.

* In the example above $$P(Y^{a=0}=1) = P(Y^{a=1}=1) = 0.5$$ and thus there is no causal effect (average).

* In our example with dichotomous variables we can re-write the Null hypothesis as $$\mathbb{E}(Y^{a=1}) = \mathbb{E}(Y^{a=0})$$.

## Section 3: Measures of causal effect 

* We introduce the following three measures of causal effect which are slighly more specific that just noting that $$P(Y^{a=1}=1) \neq P(Y^{a=0}=1)$$

 1. $$P(Y^{a=1} =1) - P(Y^{a=0}=1)$$ known as $$\textbf{causal risk}$$
 2. $$\frac{P(Y^{a=1} =1)}{P(Y^{a=0}=1)}$$ known as $$\textbf{risk ratio}$$
 3. $$\frac{P(Y^{a=1} =1)/P(Y^{a=1}=0)}{P(Y^{a=0}=1)/P(Y^{a=0}=0)}$$ known as $$\textbf{odds ratio}$$

* Equivalent ways to represent the causal null

1. $$P(Y^{a=1} =1) - P(Y^{a=0}=1) = 0$$ causal risk
2. $$\frac{P(Y^{a=1} =1)}{P(Y^{a=0}=1)} = 1$$ risk ratio
3. $$\frac{P(Y^{a=1} =1)/P(Y^{a=1}=0)}{P(Y^{a=0}=1)/P(Y^{a=0}=0)} = 1$$ odds ratio

* We can verify for our example that the risk ratio is indeed 1. 


```python
PY_a1 = data['y^a=1'].sum()/data.shape[0]
PY_a0 = data['y^a=0'].sum()/data.shape[0]

risk_ratio = PY_a1/PY_a0

if risk_ratio == 1:
    print('Null hypothesis of no causal effect is satisfied')
else:
    print('Null hypothesis of no causal effect is NOT satisfied')
```

Null hypothesis of no causal effect is satisfied


## Section 4: Summary 

* For each person in our population we have potential outcomes $$Y^{a=j}$$ which represent the outcome (in our example binary) under treatment $$j$$. We will only get to observe one of these potential outcomes. 

* For purposes of illustration we created a complete data set with no missing values to illustrate the concepts.

* We introduce a notion of random causal effect by picking individuals at random from the population. This lead to the notion of $$P(Y^{a=j}=1)$$, the probability that a randomly chosen individual will have a potential outcome equal to $$1$$ under treatment $$j$$. 

* We define three measures for average causal effect allowing us to quantify the causal difference of the two outcomes in our example. 
