# Class Balancing

This function is able to quickly balance an unbalanced dataset. Three methods of class balancing are supported:

- **Undersampling**: all classes are truncated to have the same quantity of instances as the least numerous class
- **Oversampling**: all classes have instances duplicated (with the least amount of repetition possible) until they have the same quantity of instances as the most numerous class
- **Arbitrary quantity sampling**: given a desired number of instances, each class will either be oversampled or undersampled until they reach this quantity

---

> *function* MLutils.preprocessing.**balance_dataset**(*X, y, sample_quantity='undersample', shuffle=True*)

---

### Parameters

| Parameter | Description |
|---|---|
| X : *DataFrame* | Pandas DataFrame containing the dataset's features. |
| y : *DataFrame* | Pandas DataFrame containing the dataset's labels. |
| sample_quantity : *str, int* | Indicates the sampling method. `'undersample'` or `'oversample'`<br/> can be passed. Alternatively, an integer can be passed to <br/> automatically oversample or undersample each individual <br/> class until the number of instances matches the integer. |
| shuffle : *bool* | Boolean to indicate whether the balanced dataset should be <br/> shuffled or not. |

---

### Returns

| Output | Description |
|---|---|
| X_balanced : *DataFrame* | Pandas DataFrame containing the balanced features. |
| y_balanced : *DataFrame* | Pandas DataFrame containing the balanced labels. |

---

### Example

```python
from MLutils.preprocessing import balance_dataset
import random
import pandas as pd

# create a unbalanced dummy dataset
labels = ['yes'] * 2 + ['no'] * 4 + ['maybe'] * 6 # unbalanced labels
feature_1 = [random.randrange(1, 50, 1) for f1 in range(len(labels))] # random features
feature_2 = [random.randrange(1, 50, 1) for f2 in range(len(labels))]
data = pd.DataFrame({'Feature 1': feature_1,
                     'Feature 2': feature_2,
                     'Labels': labels})

# show the dataset
print(data)
```

```python
    Feature 1  Feature 2 Labels
0          29          8    yes
1          13         49    yes
2          38         47     no
3           8         13     no
4          34         11     no
5          38         35     no
6          35         31  maybe
7          31         48  maybe
8          16         29  maybe
9          28         17  maybe
10         12         26  maybe
11         39         46  maybe
```

```python
# balance the dataset
X_balanced, y_balanced = balance_dataset(data[['Feature 1', 'Feature 2']], 
										 data['Labels'], 
										 sample_quantity=4, 
										 shuffle=False)

print(pd.concat([X_balanced, y_balanced], axis=1))
```

```python
    Feature 1  Feature 2      0
0          29          8    yes
1          13         49    yes
2          29          8    yes
3          13         49    yes
4          38         47     no
5           8         13     no
6          34         11     no
7          38         35     no
8          39         46  maybe
9          31         48  maybe
10         28         17  maybe
11         16         29  maybe
```
---
