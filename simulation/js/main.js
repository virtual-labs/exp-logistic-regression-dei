/* Main Logic for Experiment Simulation */

/* 
 * Steps Data Configuration 
 */
const stepsData = [
  {
    id: 'import_libraries',
    title: 'Importing Libraries',
    blocks: [
      {
        code: `# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from ipywidgets import interact, Dropdown
print("Libraries Imported");`,
        output: `<div class="output-success">Libraries Imported</div>`
      }
    ]
  },
  {
    id: 'reading_data',
    title: 'Loading Dataset',
    blocks: [
      {
        code: `# Read and store the Dengue dataset
data = pd.read_csv('Dengue_dataset.csv')
print("Dataset reading complete")`,
        output: `<div class="output-text">Dataset reading complete</div>`
      }
    ]
  },
  {
    id: 'data_analysis',
    title: 'Data Analysis',
    blocks: [
      {
        code: `<div class="output-success"># Display the first 5 rows of the dataset</div>
data.head()`,
        output: `<table class="data-table">
  <thead>
    <tr>
      <th>index</th>
      <th>Age</th>
      <th>Fever Days</th>
      <th>Platelets</th>
      <th>Hematocrit</th>
      <th>WBC</th>
      <th>Headache</th>
      <th>EyePain</th>
      <th>Muscle Pain</th>
      <th>Joint Pain</th>
      <th>Rash</th>
      <th>Nausea</th>
      <th>Vomiting</th>
      <th>Abdominal Pain</th>
      <th>Bleeding</th>
      <th>Lethargy</th>
      <th>Dengue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td><td>61</td><td>6</td><td>56777</td><td>49.49572857</td><td>3205</td>
      <td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>1</td>
    </tr>
    <tr>
      <td>1</td><td>24</td><td>7</td><td>61250</td><td>42.41093608</td><td>3738</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td>
      <td>0</td><td>1</td><td>1</td><td>0</td><td>1</td>
    </tr>
    <tr>
      <td>2</td><td>70</td><td>7</td><td>88034</td><td>49.66143051</td><td>3052</td>
      <td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td>
      <td>0</td><td>0</td><td>0</td><td>0</td><td>1</td>
    </tr>
    <tr>
      <td>3</td><td>30</td><td>6</td><td>55130</td><td>50.20159282</td><td>2713</td>
      <td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>1</td>
    </tr>
    <tr>
      <td>4</td><td>33</td><td>3</td><td>64346</td><td>43.46980401</td><td>4888</td>
      <td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td>
      <td>1</td><td>0</td><td>1</td><td>0</td><td>1</td>
    </tr>
  </tbody>
</table>
`,
      },
      {
        code: `<div class="output-success"># Display the last 5 rows of the dataset</div>
data.tail()`,
        output: `<table class="data-table">
  <thead>
    <tr>
      <th>index</th>
      <th>Age</th>
      <th>Fever Days</th>
      <th>Platelets</th>
      <th>Hematocrit</th>
      <th>WBC</th>
      <th>Headache</th>
      <th>EyePain</th>
      <th>Muscle Pain</th>
      <th>Joint Pain</th>
      <th>Rash</th>
      <th>Nausea</th>
      <th>Vomiting</th>
      <th>Abdominal Pain</th>
      <th>Bleeding</th>
      <th>Lethargy</th>
      <th>Dengue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4995</td><td>59</td><td>2</td><td>123791</td><td>47.18861033</td><td>4239</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
      <td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4996</td><td>11</td><td>4</td><td>111575</td><td>41.84560446</td><td>5989</td>
      <td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4997</td><td>19</td><td>2</td><td>128198</td><td>40.2842782</td><td>4775</td>
      <td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
      <td>0</td><td>0</td><td>0</td><td>1</td><td>0</td>
    </tr>
    <tr>
      <td>4998</td><td>11</td><td>0</td><td>141619</td><td>47.3939162</td><td>4625</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td>
      <td>1</td><td>1</td><td>0</td><td>1</td><td>0</td>
    </tr>
    <tr>
      <td>4999</td><td>69</td><td>2</td><td>90619</td><td>46.47148497</td><td>5982</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
  </tbody>
</table>
`,
      },
      {
        code: `<div class="output-success"># Show statistical summary of numerical columns</div>
data.describe()`,
        output: `<table class="data-table stats-table">
  <thead>
    <tr>
      <th></th>
      <th>Age</th>
      <th>Fever Days</th>
      <th>Platelets</th>
      <th>Hematocrit</th>
      <th>WBC</th>
      <th>Headache</th>
      <th>Eye Pain</th>
      <th>Muscle Pain</th>
      <th>Joint Pain</th>
      <th>Rash</th>
      <th>Nausea</th>
      <th>Vomiting</th>
      <th>Abdominal Pain</th>
      <th>Bleeding</th>
      <th>Lethargy</th>
      <th>Dengue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td>
      <td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td>
      <td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td><td>5000.0</td>
    </tr>

    <tr>
      <td>mean</td>
      <td>41.85</td><td>3.41</td><td>141522.48</td><td>41.86</td><td>6156.58</td>
      <td>0.517</td><td>0.492</td><td>0.508</td><td>0.500</td><td>0.496</td>
      <td>0.501</td><td>0.507</td><td>0.496</td><td>0.226</td><td>0.502</td><td>0.460</td>
    </tr>

    <tr>
      <td>std</td>
      <td>18.80</td><td>2.05</td><td>87116.73</td><td>4.81</td><td>2594.06</td>
      <td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td>
      <td>0.500</td><td>0.500</td><td>0.500</td><td>0.418</td><td>0.500</td><td>0.498</td>
    </tr>

    <tr>
      <td>min</td>
      <td>10.0</td><td>0.0</td><td>15060.0</td><td>32.02</td><td>2002.0</td>
      <td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td>
      <td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td>
    </tr>

    <tr>
      <td>25%</td>
      <td>26.0</td><td>2.0</td><td>62365.5</td><td>38.43</td><td>3848.75</td>
      <td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td>
      <td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td>
    </tr>

    <tr>
      <td>50%</td>
      <td>42.0</td><td>3.0</td><td>130972.0</td><td>42.02</td><td>6169.0</td>
      <td>1.0</td><td>0.0</td><td>1.0</td><td>0.5</td><td>0.0</td>
      <td>1.0</td><td>1.0</td><td>0.0</td><td>0.0</td><td>1.0</td><td>0.0</td>
    </tr>

    <tr>
      <td>75%</td>
      <td>58.0</td><td>5.0</td><td>221014.25</td><td>44.89</td><td>8302.5</td>
      <td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td>
      <td>1.0</td><td>1.0</td><td>1.0</td><td>0.0</td><td>1.0</td><td>1.0</td>
    </tr>

    <tr>
      <td>max</td>
      <td>74.0</td><td>7.0</td><td>299988.0</td><td>51.99</td><td>10993.0</td>
      <td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td>
      <td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td>
    </tr>
  </tbody>
</table>
`,
      },
      {
        code: `<div class="output-success"># Display dataset structure and data types</div>
data.info()`,
        output: `<div class="output-text">&lt;class 'pandas.core.frame.DataFrame'&gt;</div>
<div class="output-text">RangeIndex: 5000 entries, 0 to 4999</div>
<div class="output-text">Data columns (total 16 columns):</div>
<table class="data-table" style="width: auto;">
  <thead>
    <tr>
      <th>#</th>
      <th>Column</th>
      <th>Non-Null Count</th>
      <th>Dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>Age</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>1</td><td>FeverDays</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>2</td><td>Platelets</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>3</td><td>Hematocrit</td><td>5000 non-null</td><td>float64</td></tr>
    <tr><td>4</td><td>WBC</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>5</td><td>Headache</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>6</td><td>EyePain</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>7</td><td>MusclePain</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>8</td><td>JointPain</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>9</td><td>Rash</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>10</td><td>Nausea</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>11</td><td>Vomiting</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>12</td><td>AbdominalPain</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>13</td><td>Bleeding</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>14</td><td>Lethargy</td><td>5000 non-null</td><td>int64</td></tr>
    <tr><td>15</td><td>Dengue</td><td>5000 non-null</td><td>int64</td></tr>
  </tbody>
</table>
<div class="output-text" style="margin-top:5px;">dtypes: float64(1), int64(15)</div>
<div class="output-text">memory usage: 625.1 KB</div>`
      },
      {
        code: `<div class="output-success"># Check the number of missing values in each column</div>
data.isnull().sum()`,
        output: `<table class="data-table" style="width: auto;">
  <thead>
    <tr>
      <th>Column</th>
      <th>Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Age</td><td>0</td></tr>
    <tr><td>FeverDays</td><td>0</td></tr>
    <tr><td>Platelets</td><td>0</td></tr>
    <tr><td>Hematocrit</td><td>0</td></tr>
    <tr><td>WBC</td><td>0</td></tr>
    <tr><td>Headache</td><td>0</td></tr>
    <tr><td>EyePain</td><td>0</td></tr>
    <tr><td>MusclePain</td><td>0</td></tr>
    <tr><td>JointPain</td><td>0</td></tr>
    <tr><td>Rash</td><td>0</td></tr>
    <tr><td>Nausea</td><td>0</td></tr>
    <tr><td>Vomiting</td><td>0</td></tr>
    <tr><td>AbdominalPain</td><td>0</td></tr>
    <tr><td>Bleeding</td><td>0</td></tr>
    <tr><td>Lethargy</td><td>0</td></tr>
    <tr><td>Dengue</td><td>0</td></tr>
  </tbody>
</table>
<div class="output-text" style="font-size:0.8rem; margin-top:5px;">dtype: int64</div>`
      },
      {
        code: `<div class="output-success"># Show the number of rows and columns in the dataset</div>
data.shape`,
        output: `<div class="output-text">(5000, 16)</div>`
      },
      {
        code: `<div class="output-success"># Count occurrences of each class in the Dengue column</div>
data['Dengue'].value_counts()`,
        output: `<div class="output-text">0    2700</div><div class="output-text">1    2300</div><div class="output-text">Name: Dengue, dtype: int64</div>`
      },
      {
        code: `<div class="output-success"># Count frequency of each Platelets value</div>
data['Platelets'].value_counts()`,
        output: `<div class="output-text"><strong>Platelets</strong></div>

<div class="output-text">244845    2</div>
<div class="output-text">32653     2</div>
<div class="output-text">26618     2</div>
<div class="output-text">61015     2</div>
<div class="output-text">58169     2</div>

<div class="output-text">...</div>

<div class="output-text">25367     1</div>
<div class="output-text">72241     1</div>
<div class="output-text">18385     1</div>
<div class="output-text">51701     1</div>
<div class="output-text">70182     1</div>

<div class="output-text">4961 rows × 1 columns</div>
<div class="output-text">dtype: int64</div>
`,
      },
      {
        code: `<div class="output-success"># Count frequency of each WBC value</div>
data['WBC'].value_counts()`,
        output: `<div class="output-text"><strong>WBC</strong></div>

<div class="output-text">6916     6</div>
<div class="output-text">6573     5</div>
<div class="output-text">9771     5</div>
<div class="output-text">2521     5</div>
<div class="output-text">7405     5</div>

<div class="output-text">...</div>

<div class="output-text">3987     1</div>
<div class="output-text">4247     1</div>
<div class="output-text">5815     1</div>
<div class="output-text">3598     1</div>
<div class="output-text">5860     1</div>

<div class="output-text">3781 rows × 1 columns</div>
<div class="output-text">dtype: int64</div>
`,
      },
    ]
  },
  {
    id: 'data_preprocessing',
    title: 'Data Preprocessing',
    blocks: [
      {
        code: `<div class="output-success"># Encode Dengue column by flipping values (0 → 1, 1 → 0)</div>
data = data.replace({'Dengue':{0:1, 1:0}})
data.head()`,
        output: `<div class="output-text"><strong>Dataset Preview:</strong></div>

<table class="data-table">
  <thead>
    <tr>
      <th>index</th>
      <th>Age</th>
      <th>FeverDays</th>
      <th>Platelets</th>
      <th>Hematocrit</th>
      <th>WBC</th>
      <th>Headache</th>
      <th>EyePain</th>
      <th>MusclePain</th>
      <th>JointPain</th>
      <th>Rash</th>
      <th>Nausea</th>
      <th>Vomiting</th>
      <th>AbdominalPain</th>
      <th>Bleeding</th>
      <th>Lethargy</th>
      <th>Dengue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td><td>61</td><td>6</td><td>56777</td><td>49.4957</td><td>3205</td>
      <td>0</td><td>1</td><td>0</td><td>0</td><td>0</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>1</td><td>24</td><td>7</td><td>61250</td><td>42.4109</td><td>3738</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>1</td>
      <td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>2</td><td>70</td><td>7</td><td>88034</td><td>49.6614</td><td>3052</td>
      <td>1</td><td>0</td><td>1</td><td>0</td><td>0</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>3</td><td>30</td><td>6</td><td>55130</td><td>50.2016</td><td>2713</td>
      <td>1</td><td>0</td><td>0</td><td>1</td><td>0</td>
      <td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4</td><td>33</td><td>3</td><td>64346</td><td>43.4698</td><td>4888</td>
      <td>1</td><td>1</td><td>0</td><td>1</td><td>1</td>
      <td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td>
    </tr>
  </tbody>
</table>
`,
      },
      {
        code: `<div class="output-success"># Encode Headache column by flipping values (0 → 1, 1 → 0)</div>
data = data.replace({'Headache':{0:1, 1:0}})
data.head()`,
        output: `<div class="output-text"><strong>Dataset Preview:</strong></div>

<table class="data-table">
  <thead>
    <tr>
      <th>index</th>
      <th>Age</th>
      <th>FeverDays</th>
      <th>Platelets</th>
      <th>Hematocrit</th>
      <th>WBC</th>
      <th>Headache</th>
      <th>EyePain</th>
      <th>MusclePain</th>
      <th>JointPain</th>
      <th>Rash</th>
      <th>Nausea</th>
      <th>Vomiting</th>
      <th>AbdominalPain</th>
      <th>Bleeding</th>
      <th>Lethargy</th>
      <th>Dengue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td><td>61</td><td>6</td><td>56777</td><td>49.4957</td><td>3205</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>1</td><td>24</td><td>7</td><td>61250</td><td>42.4109</td><td>3738</td>
      <td>0</td><td>0</td><td>0</td><td>0</td><td>1</td>
      <td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>2</td><td>70</td><td>7</td><td>88034</td><td>49.6614</td><td>3052</td>
      <td>0</td><td>0</td><td>1</td><td>0</td><td>0</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>3</td><td>30</td><td>6</td><td>55130</td><td>50.2016</td><td>2713</td>
      <td>0</td><td>0</td><td>0</td><td>1</td><td>0</td>
      <td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4</td><td>33</td><td>3</td><td>64346</td><td>43.4698</td><td>4888</td>
      <td>0</td><td>1</td><td>0</td><td>1</td><td>1</td>
      <td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td>
    </tr>
  </tbody>
</table>
`,
      },
      {
        code: `<div class="output-success"># Encode Rash column by flipping values (0 → 1, 1 → 0) and verify changes</div>
data = data.replace({'Rash':{0:1, 1:0}})
print("Encoding complete. Checked head:")
data.head()`,
        output: `<div class="output-text"><div class="output-success">Encoding complete. Checked head:</div>
<strong>Dataset Preview:</strong></div>

<table class="data-table">
  <thead>
    <tr>
      <th>index</th>
      <th>Age</th>
      <th>FeverDays</th>
      <th>Platelets</th>
      <th>Hematocrit</th>
      <th>WBC</th>
      <th>Headache</th>
      <th>EyePain</th>
      <th>MusclePain</th>
      <th>JointPain</th>
      <th>Rash</th>
      <th>Nausea</th>
      <th>Vomiting</th>
      <th>AbdominalPain</th>
      <th>Bleeding</th>
      <th>Lethargy</th>
      <th>Dengue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td><td>61</td><td>6</td><td>56777</td><td>49.4957</td><td>3205</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>1</td>
      <td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>1</td><td>24</td><td>7</td><td>61250</td><td>42.4109</td><td>3738</td>
      <td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
      <td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>2</td><td>70</td><td>7</td><td>88034</td><td>49.6614</td><td>3052</td>
      <td>0</td><td>0</td><td>1</td><td>0</td><td>1</td>
      <td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>3</td><td>30</td><td>6</td><td>55130</td><td>50.2016</td><td>2713</td>
      <td>0</td><td>0</td><td>0</td><td>1</td><td>1</td>
      <td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4</td><td>33</td><td>3</td><td>64346</td><td>43.4698</td><td>4888</td>
      <td>0</td><td>1</td><td>0</td><td>1</td><td>0</td>
      <td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td>
    </tr>
  </tbody>
</table>
`,

      },

    ]
  },

  {
    id: 'model_training',
    title: 'Model Training',
    blocks: [
      {
        code: `<div class="output-success"># Set the target variable, separate features and labels, and split the dataset into 80% training and 20% testing data</div>
target_col = "Dengue"
print("Target column set to:", target_col)
X = data.drop(columns=['Platelets', target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Split: 80% Train ({len(X_train)}) / 20% Test ({len(X_test)}) samples")`,
        output: `<div class="output-text">Target column set to: Dengue</div>
<div class="output-text">Split: 80% Train (4000) / 20% Test (1000) samples</div>`
      },
      {
        code: `<div class="output-success"># Display the feature matrix</div>
X`,
        output: `<div class="output-text"><strong>Feature Matrix (X) Preview:</strong></div>
<table class="data-table">
  <thead>
    <tr>
      <th></th><th>Age</th><th>FeverDays</th><th>Platelets</th><th>Hematocrit</th><th>WBC</th><th>Headache</th><th>EyePain</th><th>MusclePain</th><th>JointPain</th><th>Rash</th><th>Nausea</th><th>Vomiting</th><th>AbdominalPain</th><th>Bleeding</th><th>Lethargy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td><td>61</td><td>6</td><td>56777</td><td>49.495729</td><td>3205</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>1</td><td>24</td><td>7</td><td>61250</td><td>42.410936</td><td>3738</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td>
    </tr>
    <tr>
      <td>2</td><td>70</td><td>7</td><td>88034</td><td>49.661431</td><td>3052</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>3</td><td>30</td><td>6</td><td>55130</td><td>50.201593</td><td>2713</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4</td><td>33</td><td>3</td><td>64346</td><td>43.469804</td><td>4888</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td>
    </tr>
    <tr>
      <td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>
    </tr>
    <tr>
      <td>4995</td><td>59</td><td>2</td><td>123791</td><td>47.188610</td><td>4239</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4996</td><td>11</td><td>4</td><td>111575</td><td>41.845604</td><td>5989</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td>
    </tr>
    <tr>
      <td>4997</td><td>19</td><td>2</td><td>128198</td><td>40.284278</td><td>4775</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td>
    </tr>
    <tr>
      <td>4998</td><td>11</td><td>0</td><td>141619</td><td>47.393916</td><td>4625</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td>
    </tr>
    <tr>
      <td>4999</td><td>69</td><td>2</td><td>90619</td><td>46.471485</td><td>5982</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td>
    </tr>
  </tbody>
</table>
<div class="output-text" style="font-size:0.8rem; margin-top:5px;">5000 rows × 15 columns</div>`
      },
      {
        code: `<div class="output-success"># Display the first few values of the target variable</div>
y.head()`,
        output: `<div class="output-text"><strong>Target Vector (y) Preview:</strong></div>
<table class="data-table" style="max-width: 150px;">
  <thead>
    <tr><th></th><th>Dengue</th></tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>0</td></tr>
    <tr><td>1</td><td>0</td></tr>
    <tr><td>2</td><td>0</td></tr>
    <tr><td>3</td><td>0</td></tr>
    <tr><td>4</td><td>0</td></tr>
  </tbody>
</table>
<div class="output-text" style="font-size:0.8rem; margin-top:5px;">Name: Dengue, dtype: int64</div>`
      },
      {
        code: `<div class="output-success"># Initialize and train a Logistic Regression model with L2 regularization on the training data</div>
model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
model.fit(X_train, Y_train)
print("Model trained successfully")`,
        output: `<img src="./images/model_output.png" alt="Sklearn Output" style="max-width: 30%; border: 1px solid #ddd; border-radius: 4px;">
</div>
<div class="output-success">Model trained successfully</div>
<div style="margin-top: 10px;">`
      }
    ]
  },
  {
    id: 'model_evaluation',
    title: 'Model Evaluation',
    blocks: [
      {
        code: `<div class="output-success"># Preparing Feature and Target Data for Logistic Regression Analysis</div>
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
feature_names = numeric_cols
X_num = X[feature_names]
y_arr = np.array(y)
print(f"Feature-target data ready → {len(feature_names)} numeric features selected for modeling")`,
        output: `<div class="output-text">Feature-target data ready → 14 numeric features selected for modeling</div>`
      },
      {
        code: `<div class="output-success"># Fits a logistic regression model on a single feature and visualizes the resulting sigmoid probability curve against the actual data.</div>
def show_1d_sigmoid(f):
    X,y = X_num[[f]].values, y_arr; c = LogisticRegression(max_iter=5000).fit(X,y)
    a,b = X.min(),X.max(); 
    a,b = (a-1,b+1) if a==b else (a,b); g= np.linspace(a,b,300)[:,None]; p = c.predict_proba(g)[:,1]
    plt.scatter(X[y==0],y[y==0]); plt.scatter(X[y==1],y[y==1]); plt.plot(g,p); plt.xlabel(f); 
    plt.ylabel("Probability of Dengue");
    plt.ylim(-.1,1.1); plt.grid(1); plt.title(f"Sigmoid using {f}");
    plt.show()

print("Function defined: show_1d_sigmoid")`,
        output: `<div class="output-success">Function defined: show_1d_sigmoid</div>`
      },
      {
        code: `<div class="output-success"># Interactive dropdown over all features</div>
interact(
    show_1d_sigmoid,
    feat=Dropdown(options=feature_names, description="Feature")
)`,
        output: `<div style="padding:0px;">
    <!-- Standard layout -->
    <div style="margin-bottom:5px; display:flex; align-items:center;">
        <label style="font-family:sans-serif; margin-right:10px; font-weight:normal; font-size: 1rem;">Feature</label>
        <select id="featureSelect" onchange="window.updateSigmoidPlot && window.updateSigmoidPlot()" style="padding:2px 5px; border:1px solid #ccc; border-radius:3px; outline:none; margin-bottom:0;">
            <option value="Age">Age</option>
            <option value="FeverDays">FeverDays</option>
            <option value="Platelets" selected>Platelets</option>
            <option value="Hematocrit">Hematocrit</option>
            <option value="WBC">WBC</option>
            <option value="MusclePain">MusclePain</option>
            <option value="JointPain">JointPain</option>
            <option value="Nausea">Nausea</option>
            <option value="Vomiting">Vomiting</option>
            <option value="AbdominalPain">AbdominalPain</option>
            <option value="Bleeding">Bleeding</option>
            <option value="Lethargy">Lethargy</option>
            <option value="Rash">Rash</option>
            <option value="EyePain">EyePain</option>
            <option value="Headache">Headache</option>
        </select>
    </div>
    <!-- No white background on container, align left (default) -->
    <div id="sigmoidPlot" style="text-align:left; margin-top:10px;">
       <img src="./images/Platelets.png" alt="Sigmoid of Platelets" style="max-height:300px; border:none; display:block;">
    </div>
</div>`
      },
      {
        code: `<div class="output-success"># Training Performance</div>
Y_train_pred = model.predict(X_train)
print(" Training Performance ")
print(f"Accuracy : {accuracy_score(y_train, Y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, Y_train_pred):.4f}")
print(f"Recall   : {recall_score(y_train, Y_train_pred):.4f}")
print(f"F1 Score : {f1_score(y_train, Y_train_pred):.4f}")`,
        output: `<div class="output-header">Training Performance</div>
<div class="output-text">Accuracy : 0.9732</div>
<div class="output-text">Precision: 0.9712</div>
<div class="output-text">Recall   : 0.9707</div>
<div class="output-text">F1 Score : 0.9709</div>`
      },
      {
        code: `<div class="output-success"># Confusion Matrix for training data</div>
cm_train = confusion_matrix(y_train, Y_train_pred)
sns.heatmap(cm_train, cmap="Blues")
plt.title("Confusion Matrix (Training Data)")
for i in range(len(cm_train)):
    for j in range(len(cm_train)):
        plt.text(j, i, cm_train[i][j], ha='center', va='center')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
        output: `<div style="text-align:left; padding:0px;">
    <h4 style="margin:0 0 10px 0;">Confusion Matrix (Training Data)</h4>
    <div id="confusionMatContainer" style="display:flex; justify-content:flex-start; align-items:flex-start; padding-top:10px;">
        <img src="./images/CM(TD).png" style="max-width:100%; max-height:300px; border: 1px solid #ddd; border-radius: 4px;" alt="Confusion Matrix">
    </div>
</div>`
      },
      {
        code: `<div class="output-success"># Testing Performance</div>
Y_test_pred = model.predict(X_test)
print("Testing Performance")
print(f"Accuracy : {accuracy_score(y_test, Y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, Y_test_pred):.4f}")
print(f"Recall   : {recall_score(y_test, Y_test_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, Y_test_pred):.4f}")`,
        output: `<div class="output-header">Testing Performance</div>
<div class="output-text">Accuracy : 0.9740</div>
<div class="output-text">Precision: 0.9657</div>
<div class="output-text">Recall   : 0.9783</div>
<div class="output-text">F1 Score : 0.9719</div>`
      },
      {
        code: `<div class="output-success"># Confusion Matrix for testing data</div>
cm_test = confusion_matrix(y_test, Y_test_pred)
sns.heatmap(cm_test, annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Testing Data")
plt.show()`,
        output: `<div style="text-align:left; padding:0px;">
    <h4 style="margin:0 0 10px 0;">Confusion Matrix (Testing Data)</h4>
    <div id="confusionMatContainer" style="display:flex; justify-content:flex-start; align-items:flex-start; padding-top:10px;">
        <img src="./images/CM.png" style="max-width:100%; max-height:300px; border: 1px solid #ddd; border-radius: 4px;" alt="Confusion Matrix">
    </div>
</div>`
      },
      {
        code: `<div class="output-success"># Calculates predicted probabilities to compute ROC curve metrics and the AUC score.</div>
Y_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, th = roc_curve(Y_test, Y_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC metrics calculated. AUC = {roc_auc:.3f}")`,
        output: `<div class="output-success">ROC metrics calculated. AUC = 0.998</div>`
      },
      {
        code: `<div class="output-success"># Plots the ROC curve including the AUC score and a random classifier baseline.</div>
plt.plot(fpr, tpr, linewidth=3, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()`,
        output: `<div style="text-align:left; padding:0px;">
    <h4 style="margin:0 0 10px 0;">ROC Curve</h4>
    <div style="display:flex; justify-content:flex-start;margin-top:10px; align-items:flex-start; padding-top:0;">
        <img src="./images/roc.png" style="max-width:100%; max-height:300px; margin-top:0px;" alt="ROC Curve">
    </div>
</div>`
      },
      {
        code: `<div class="output-success"># Random Test Prediction</div>
test_samples=data.sample(5)
display(test_samples)
sample = X_test.sample(1)
actual = Y_test.loc[sample.index].values[0]
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]
print("------ Random Sample Test ------")
print(sample)
print("Actual Dengue:", actual)
print("Predicted Dengue:", pred)
print("Probability:", prob)`,
        output: `<div id="randomPredContainer" style="font-family:monospace; font-size:0.9em; padding:5px;">
    <div style="margin-bottom:10px; font-weight:bold; color:#555;">Select a row to see prediction:</div>
    <div style="overflow-x:auto; margin-bottom:5px;">
        <table class="data-table" style="cursor:pointer; width:100%;">
            <thead>
                <tr>
                    <th>Index</th><th>Age</th><th>FeverDays</th><th>Hematocrit</th><th>WBC</th><th>Headache</th><th>EyePain</th><th>MusclePain</th><th>JointPain</th><th>Rash</th><th>Nausea</th><th>Vomiting</th><th>AbdominalPain</th><th>Bleeding</th><th>Lethargy</th>
                </tr>
            </thead>
            <tbody id="randomPredTableBody">
               <!-- Rows injected by JS -->
            </tbody>
        </table>
    </div>
    
    <!-- Result box: Initially hidden, styled like a code block (Pandas output) -->
    <div id="randomPredResult" style="padding:15px; display:none; white-space: pre; overflow-x: auto; font-family: monospace; color: #333;">
        <!-- Result injected here -->
    </div>
</div>`
      }

    ]

  }
];

// State Management
let STATE = {
  stepIndex: 0,
  subStepIndex: 0,
  stepsStatus: stepsData.map(() => ({ unlocked: false, completed: false, partial: false }))
};

// Initial State: First step unlocked
STATE.stepsStatus[0].unlocked = true;

// DOM Elements
const stepsContainer = document.getElementById('stepsContainer');
const codeDisplay = document.getElementById('codeDisplay');
const outputDisplay = document.getElementById('outputDisplay');
const outputContent = document.getElementById('outputDisplay'); // Wrapper reuse
const bottomPane = document.querySelector('.bottom-pane');
const runBtn = document.getElementById('runBtn');

// Initialize UI
function init() {
  renderSidebar();
  loadStep(0);
}

// Render Sidebar with Color Logic
function renderSidebar() {
  stepsContainer.innerHTML = '';

  stepsData.forEach((step, index) => {
    const status = STATE.stepsStatus[index];
    const btn = document.createElement('button');
    btn.classList.add('step-btn');
    btn.innerText = step.title;

    // Label Logic
    let label = `${index + 1}. ${step.title}`;
    if (status.completed) label = `✓ ${step.title}`;
    btn.innerText = label;

    // Styling & Interaction Logic
    if (status.unlocked) {
      if (status.completed) {
        btn.classList.add('completed');
      } else if (status.partial) {
        btn.classList.add('in-progress');
      } else {
        // Default unlocked state (Blue via CSS)
      }

      btn.disabled = false;
      btn.style.cursor = 'pointer';

      // Mark current active
      if (index === STATE.stepIndex) {
        btn.classList.add('active');
      }

      // Allow click to load/revisit
      btn.onclick = () => {
        loadStep(index);
      };
    } else {
      // Locked -> Grey
      btn.classList.add('disabled');
      btn.innerText = label;
      btn.disabled = true;
      btn.onclick = (e) => e.preventDefault();
    }

    stepsContainer.appendChild(btn);
  });

  // Loading Spinner (Hidden by default)
  const loader = document.createElement('div');
  loader.className = 'loading-spinner';
  loader.innerText = 'Loading...';
  loader.style.width = '100%';
  loader.style.textAlign = 'center';
  loader.style.marginTop = '20px';
  loader.style.display = 'none';
  stepsContainer.appendChild(loader);

  // Add Restart Button at the end
  const restartBtn = document.createElement('button');
  restartBtn.classList.add('step-btn');
  restartBtn.innerText = "Restart Experiment";
  restartBtn.style.backgroundColor = "#333";
  restartBtn.style.textAlign = 'center';
  restartBtn.style.marginTop = "auto";
  restartBtn.style.color = "white";
  restartBtn.onclick = restartExperiment;
  stepsContainer.appendChild(restartBtn);

  // Add Download Button below Restart
  const downloadBtn = document.createElement('button');
  downloadBtn.classList.add('step-btn');
  downloadBtn.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
    Download Experiment
  `;
  downloadBtn.style.backgroundColor = "#F57C2A"; // Orange (#F57C2A)
  downloadBtn.style.textAlign = 'center';
  downloadBtn.style.marginTop = "10px";
  downloadBtn.style.color = "white";
  downloadBtn.onclick = downloadPDF;
  stepsContainer.appendChild(downloadBtn);
}


function loadStep(index) {
  STATE.stepIndex = index;
  STATE.subStepIndex = 0; // Fix: Always reset sub-step when loading a main step
  renderSidebar();
  updateUI();
}

function updateUI() {
  const step = stepsData[STATE.stepIndex];
  if (STATE.subStepIndex >= step.blocks.length) STATE.subStepIndex = 0;

  const block = step.blocks[STATE.subStepIndex];

  // Extract and Update Comment Header
  const commentMatch = block.code.match(/#\s*([^<\n\r]*)/); // Extract comment until tag or newline
  const codeHeaderBar = document.getElementById('codeHeaderBar');
  if (commentMatch) {
    codeHeaderBar.innerText = "# " + commentMatch[1].trim();
    codeHeaderBar.style.display = 'block';
  } else {
    codeHeaderBar.style.display = 'none';
  }

  // Update Code (Remove all HTML tags and then extract code)
  const codeWithoutTags = block.code.replace(/<[^>]*>/g, '');
  const codeWithoutComment = codeWithoutTags.replace(/#\s*.*/, '').trim();
  codeDisplay.innerHTML = highlightCode(codeWithoutComment);

  // Reset Output
  bottomPane.classList.remove('active-output');
  // Reset any inline styles added by completion message
  bottomPane.style.display = '';
  bottomPane.style.flexDirection = '';
  bottomPane.style.justifyContent = '';
  bottomPane.style.alignItems = '';

  outputContent.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';

  // Reset Button State (Simple & Safe)
  runBtn.style.display = 'flex'; // Fix: Ensure button is visible after restart
  runBtn.classList.remove('completed');
  runBtn.style.backgroundColor = '#F57C2A'; // Orange (#F57C2A)
  runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
  runBtn.disabled = false;

  // ALWAYS reset onclick to standard runStep
  runBtn.onclick = runStep;
}

function runStep() {
  const step = stepsData[STATE.stepIndex];
  const block = step.blocks[STATE.subStepIndex];

  // 1. Loading State
  outputContent.innerHTML = '<div class="loading-spinner">Running code...</div>';
  runBtn.disabled = true;

  // 2. Simulated Delay (2 seconds)
  setTimeout(() => {
    // 3. Show Output
    outputContent.innerHTML = block.output;
    bottomPane.classList.add('active-output');

    // 4. Update Button State to Checkmark (Success)
    runBtn.classList.add('completed');
    runBtn.style.backgroundColor = '#A6CE63'; // Green (#A6CE63)
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

    // Mark partial progress
    STATE.stepsStatus[STATE.stepIndex].partial = true;
    renderSidebar();

    // Check if this is the Random Prediction block
    if (document.getElementById('randomPredTableBody')) {
      window.generateRandomPrediction && window.generateRandomPrediction();
    }

    // 5. Handle Next Logic
    const hasNextBlock = STATE.subStepIndex < step.blocks.length - 1;

    if (hasNextBlock) {
      // Wait 1s then change button to "Next"
      setTimeout(() => {
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
        runBtn.style.backgroundColor = '#5FA8E4'; // Orange
        runBtn.disabled = false;

        // Switch handler to Next
        runBtn.onclick = nextSubStep;
      }, 500);

    } else {
      // Step Fully Completed
      STATE.stepsStatus[STATE.stepIndex].completed = true;
      renderSidebar(); // Update Current Step to Green Immediately

      // Unlock next step logic
      if (STATE.stepIndex < stepsData.length - 1) {
        STATE.stepsStatus[STATE.stepIndex + 1].unlocked = true;
        renderSidebar(); // Update Next Step to Red Immediately

        // Manual Next Step Arrow Button
        setTimeout(() => {
          // Change button to Blue Arrow for Next Step
          runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
          runBtn.style.backgroundColor = '#5FA8E4'; // Blue (#5FA8E4)
          runBtn.disabled = false;

          // Logic to go to next MAIN step
          runBtn.onclick = function () {
            loadStep(STATE.stepIndex + 1);
          };
        }, 500);
      } else {
        // End of Experiment - Show "Next" (Finish) Button
        renderSidebar();
        setTimeout(() => {
          runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
          runBtn.style.backgroundColor = '#72b2f7ff'; // Orange
          runBtn.disabled = false;
          runBtn.onclick = showCompletionMessage;
        }, 500);
      }
    }

  }, 500);
}

function nextSubStep() {
  STATE.subStepIndex++;
  updateUI(); // This will reset button to Red/Run for the new block
}

function restartExperiment() {
  // Reset State
  STATE.stepIndex = 0;
  STATE.subStepIndex = 0;
  STATE.stepsStatus = stepsData.map(() => ({ unlocked: false, completed: false, partial: false }));
  STATE.stepsStatus[0].unlocked = true;

  init();
}

function highlightCode(code) {
  return code
    .replace(/import /g, '<span class="kw">import </span>')
    .replace(/from /g, '<span class="kw">from </span>')
    .replace(/print/g, '<span class="func">print</span>')
    .replace(/def /g, '<span class="kw">def </span>')
    .replace(/return /g, '<span class="kw">return </span>');
}

// Global scope for HTML callbacks
window.updateSigmoidPlot = function () {
  const featureSelect = document.getElementById('featureSelect');
  if (!featureSelect) return; // Guard

  const feature = featureSelect.value;
  const container = document.getElementById('sigmoidPlot');

  // Use static image matching the feature name
  // Default to Age if feature is just "Feature" or empty, but here we read value
  // Ensure the image fits comfortably without massive white borders
  container.innerHTML = `<img src="./images/${feature}.png" alt="Sigmoid of ${feature}" style="max-height:300px; border:none; display:block;">`;
};

// Global scope for Random Prediction Table
window.generateRandomPrediction = function () {
  const tbody = document.getElementById('randomPredTableBody');
  if (!tbody) return;

  // Fixed samples matching user screenshot
  const samples = [
    { index: 1434, age: 69, fever: 3, platelets: 32794, hemo: 44.363337, wbc: 4203, headache: 1, eye: 1, muscle: 1, joint: 0, rash: 0, nausea: 0, vomit: 1, abdom: 1, bleed: 0, lethargy: 0, dengue: 1, pred: 1, prob: 0.9995 },
    { index: 1435, age: 23, fever: 6, platelets: 34881, hemo: 42.601771, wbc: 2706, headache: 1, eye: 1, muscle: 0, joint: 1, rash: 0, nausea: 0, vomit: 1, abdom: 0, bleed: 1, lethargy: 1, dengue: 1, pred: 1, prob: 0.9782 },
    { index: 1436, age: 11, fever: 7, platelets: 64703, hemo: 48.035335, wbc: 3961, headache: 0, eye: 0, muscle: 1, joint: 0, rash: 1, nausea: 0, vomit: 0, abdom: 1, bleed: 0, lethargy: 1, dengue: 1, pred: 1, prob: 0.9810 },
    { index: 1437, age: 26, fever: 4, platelets: 59354, hemo: 51.648016, wbc: 4519, headache: 0, eye: 1, muscle: 0, joint: 1, rash: 0, nausea: 1, vomit: 0, abdom: 0, bleed: 1, lethargy: 0, dengue: 1, pred: 1, prob: 0.9755 },
    { index: 1438, age: 37, fever: 3, platelets: 29390, hemo: 42.108583, wbc: 4666, headache: 1, eye: 1, muscle: 0, joint: 0, rash: 0, nausea: 1, vomit: 0, abdom: 0, bleed: 1, lethargy: 1, dengue: 1, pred: 1, prob: 0.9991 }
  ];

  // Clear existing
  tbody.innerHTML = '';

  samples.forEach(s => {
    const tr = document.createElement('tr');
    tr.style.transition = 'background 0.2s';
    tr.onmouseover = () => { if (!tr.classList.contains('selected-row')) tr.style.background = '#e3f2fd'; };
    tr.onmouseout = () => { if (!tr.classList.contains('selected-row')) tr.style.background = 'white'; };
    tr.onclick = () => window.showPredictionResult(s, tr);

    tr.innerHTML = `
            <td><strong>${s.index}</strong></td>
            <td>${s.age}</td><td>${s.fever}</td><td>${s.hemo.toFixed(6)}</td><td>${s.wbc}</td>
            <td>${s.headache}</td><td>${s.eye}</td><td>${s.muscle}</td><td>${s.joint}</td>
            <td>${s.rash}</td><td>${s.nausea}</td><td>${s.vomit}</td><td>${s.abdom}</td>
            <td>${s.bleed}</td><td>${s.lethargy}</td>
        `;
    tbody.appendChild(tr);
  });
};

window.showPredictionResult = function (s, tr) {
  try {
    const resDiv = document.getElementById('randomPredResult');
    if (!resDiv) {
      console.error("randomPredResult div not found");
      return;
    }

    // Highlight selected row (reset others)
    const rows = document.querySelectorAll('#randomPredTableBody tr');
    rows.forEach(r => {
      r.style.backgroundColor = 'white';
      r.classList.remove('selected-row');
    });

    // Set Green highlight for selected
    tr.style.backgroundColor = '#a5d6a7';
    tr.classList.add('selected-row');

    // Make visible
    resDiv.style.display = 'block';

    // Format Output similar to Pandas Dataframe (Right Aligned)
    // define column widths
    const wIndex = 6;
    const wAge = 5;
    const wFever = 10;
    const wHemo = 12;
    const wWBC = 6;
    const wHead = 9;
    const wEye = 8;
    const wMus = 11;
    const wJoint = 10;
    const wRash = 5;
    const wNau = 8; // Increased slightly
    const wVom = 9;
    const wAbd = 14;
    const wBleed = 9;
    const wLeth = 9;

    // Helper for center alignment
    const center = (str, w) => {
      const s = (str === undefined || str === null) ? '' : str.toString();
      if (s.length >= w) return s;
      const totalPad = w - s.length;
      const leftPad = Math.floor(totalPad / 2);
      const rightPad = totalPad - leftPad;
      return ' '.repeat(leftPad) + s + ' '.repeat(rightPad);
    };

    const sep = '   '; // Wider gap between columns

    // Construct strings
    // Row 1 Headers
    let row1 = center('', wIndex) +
      center('Age', wAge) + sep +
      center('FeverDays', wFever) + sep +
      center('Hematocrit', wHemo) + sep +
      center('WBC', wWBC) + sep +
      center('Headache', wHead) + sep +
      center('EyePain', wEye) + sep +
      center('MusclePain', wMus); // Removed backslash

    // Row 2 Data
    let row2 = center(s.index, wIndex) +
      center(s.age, wAge) + sep +
      center(s.fever, wFever) + sep +
      center(s.hemo !== undefined ? s.hemo.toFixed(6) : '', wHemo) + sep +
      center(s.wbc, wWBC) + sep +
      center(s.headache, wHead) + sep +
      center(s.eye, wEye) + sep +
      center(s.muscle, wMus);

    // Row 3 Headers
    let row3 = center('', wIndex) +
      center('JointPain', wJoint) + sep +
      center('Rash', wRash) + sep +
      center('Nausea', wNau) + sep +
      center('Vomiting', wVom) + sep +
      center('AbdominalPain', wAbd) + sep +
      center('Bleeding', wBleed) + sep +
      center('Lethargy', wLeth);

    // Row 4 Data
    let row4 = center(s.index, wIndex) +
      center(s.joint, wJoint) + sep +
      center(s.rash, wRash) + sep +
      center(s.nausea, wNau) + sep +
      center(s.vomit, wVom) + sep +
      center(s.abdom, wAbd) + sep +
      center(s.bleed, wBleed) + sep +
      center(s.lethargy, wLeth); // Fixed typo from s.leth to s.lethargy


    resDiv.innerHTML = `
<div style="font-weight:bold; margin-bottom:5px;">------ Random Sample Test ------</div>
${row1}
${row2}

${row3}
${row4}

Actual Dengue: ${s.dengue}
Predicted Dengue: ${s.pred}
Probability: ${s.prob}
    `.trim();

  } catch (e) {
    console.error("Error in showPredictionResult:", e);
  }
};

// Global scope for Confusion Matrix Animation (Simplified for Image)
window.animateConfusionMatrix = function () {
  // No animation needed for static image, but keeping function to prevent errors if called
};

// Completion Message
function showCompletionMessage() {
  outputContent.innerHTML = ''; // Clear output content
  bottomPane.classList.add('active-output');
  bottomPane.style.display = 'flex';
  bottomPane.style.flexDirection = 'column';
  bottomPane.style.justifyContent = 'center';
  bottomPane.style.alignItems = 'center';

  const msgHTML = `
    <div style="text-align: center; animation: fadeIn 1s ease;">
      <h1 style="color: #2a9d8f; font-size: 2.5rem; margin-bottom: 20px;">Experiment Completed! ✔️</h1>
      <p style="font-size: 1.5rem; color: #333;">You have completed logistic regression successfully!</p>
      <button onclick="location.reload()" style="margin-top: 30px; padding: 15px 30px; background-color: #f7a072; color: white; border: none; border-radius: 10px; font-size: 1.2rem; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">Restart Experiment</button>
    </div>
  `;
  outputContent.innerHTML = msgHTML;
  // Hide run button or make it inactive
  runBtn.style.display = 'none';
}

// PDF Download Logic
function downloadPDF() {
  const link = document.createElement('a');
  link.href = './Exp-Logistic_Regression.pdf';
  link.download = 'Exp-Logistic_Regression.pdf';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}


// Init
function init() {
  renderSidebar();
  loadStep(0);

  // Attach Download Listener
  const downloadBtn = document.querySelector('.download-btn');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', downloadPDF);
  }
}

init();
