# How much structure can clustering find without labels?

**Final project, Option 3: Machine Learning with Python**

**Author:** Andreou Maria-Gavriela

In this project I took 210 wheat kernels, hid which variety each one belonged to, and asked a simple question: if a clustering algorithm only sees seven measurements of each kernel, how much of the real grouping can it find on its own? I kept the true varieties locked away until the very end and used them only as an answer key.

What I liked about this option is that it is honest by design. The answer exists, but the method never gets to see it, so the final check really tests what clustering can and cannot do.

## At a glance

| Question | My answer |
|---|---|
| How many groups do the measurements support? | **3**, chosen without looking at the labels |
| How many kernels landed with their own variety? | **193 of 210** (purity **0.919**) |
| Is that better than chance? | Yes, by far: adjusted Rand index **0.773**, NMI **0.728** |
| What is realistically possible? | A model that *is* allowed to see the labels gets **96.2%** right |
| Where does it go wrong? | All **17** mistakes involve one variety, **Kama** |

## Files in this repository

| File | What it is |
|---|---|
| [`Final_Project_Option3_Seeds_Clustering.ipynb`](Final_Project_Option3_Seeds_Clustering.ipynb) | My completed notebook, with every cell run and all outputs visible |
| [`report/Final_Project_Option3_Seeds_Clustering.pdf`](report/Final_Project_Option3_Seeds_Clustering.pdf) | The same notebook as a PDF: code, outputs and written answers |
| [`report/Final_Project_Option3_Seeds_Clustering.html`](report/Final_Project_Option3_Seeds_Clustering.html) | The same notebook as a web page |
| [`index.html`](index.html) | A one-page visual summary of the project (the project website) |
| [`figures/`](figures) | Every chart, saved directly by the notebook |
| [`data/seeds.csv`](data/seeds.csv) | A copy of the data, so the notebook also runs offline |

---

## 1. The problem

The scenario is an agricultural co-op that receives deliveries of mixed wheat grain with no labels. Before paying for a sorting process, the co-op wants to know two things:

- **How many varieties are plausibly in the mix?**
- **How cleanly do they separate on a few cheap measurements?**

Because the deliveries carry no labels, this is an **unsupervised** problem. I first look for groups in the measurements alone. Only after the clustering is fixed do I bring back the true varieties and check how well my groups match them.

## 2. The data

I used the **Seeds** data set from the UCI Machine Learning Repository (Charytanowicz et al., 2010). It contains **210 kernels**, 70 from each of three wheat varieties: **Kama, Rosa and Canadian**. The grain came from experimental fields at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin. Each kernel was photographed with a soft X-ray technique, and seven measurements were taken from the image:

| Feature | What it measures |
|---|---|
| `area`, `perimeter` | The size of the kernel's outline |
| `compactness` | How close the outline is to a circle (4πA / P²) |
| `kernel_length`, `kernel_width` | The kernel's long and short axes |
| `asymmetry` | How lopsided the shape is |
| `groove_length` | The length of the groove along the kernel |

All seven are numeric and nothing is missing. The notebook downloads the data from OpenML, as the course intended. I also keep a copy in `data/seeds.csv`, which the notebook uses when it is present so it runs offline. I checked that this copy matches a second, independent copy of the UCI file exactly.

## 3. My approach

1. **Scale the features.** Clustering works with distances, and the raw features sit on very different scales: area is around 10–20, compactness around 0.8–0.9. Without scaling, area would decide everything. I used `StandardScaler` so every feature counts equally.
2. **Look at the data without labels.** I used PCA to squeeze the seven features into two for plotting. Those two components keep **89%** of the variation. The first one (72%) turned out to be a *size* axis. The second (17%) is a *shape* axis that contrasts asymmetry with compactness.
3. **Choose the number of groups without labels.** I fitted K-Means for k = 2 to 7 and compared the two methods from the course, the **elbow** and the **silhouette**. When they disagreed, I added two more checks: the **gap statistic** and **bootstrap stability**.
4. **Fit the final model.** K-Means with **k = 3**, using all seven scaled features (`n_init=10`, `random_state=42`).
5. **Only then, reveal the labels.** I compared clusters with varieties using a cross-tabulation and the **purity** score. I added per-variety recall and precision, plus two chance-corrected scores (**ARI** and **NMI**).
6. **Understand the result.** I described each cluster in real units and looked closely at the kernels that ended up in the "wrong" group.
7. **Stress-test it.** I re-ran the task with two other algorithms and 25 random seeds. I compared it with supervised models that see the labels. I also tested one idea for a higher score, which I ended up rejecting.

## 4. What I found

### How many groups? The two course methods disagreed

![Elbow and silhouette](figures/02_elbow_silhouette.png)

This was the first real decision of the project, and the two methods from the course pointed in different directions:

- **The elbow said 3.** Inertia drops by 229 going from 2 to 3 clusters, but only by 59 going from 3 to 4.
- **The silhouette said 2.** It is highest at k = 2 (0.466), with k = 3 close behind (0.401). The 2-cluster solution simply merges two neighbouring groups into one of 133 kernels.

I did not want to break the tie by peeking at the labels, so I added two more label-free checks:

![Gap statistic and bootstrap stability](figures/06_gap_stability.png)

- **The gap statistic** compares my data with random data that has no groups at all. It picks **k = 3**.
- **Bootstrap stability** refits K-Means on 50 resampled versions of the data and checks whether it splits the kernels the same way each time. Three clusters are about as stable as two (mean ARI 0.88 against 0.92). From four clusters onwards the groups start to fall apart (mean ARI 0.75 or lower).

So I chose **k = 3**, and only afterwards opened the answer key.

### How well do the groups match the real varieties?

<img src="figures/03_crosstab_heatmap.png" width="420" alt="Cross-tabulation of clusters against varieties">

Each cluster is clearly dominated by one variety:

| Variety | Found in its own cluster | Recall | Precision of its cluster |
|---|---|---|---|
| Rosa | 65 / 70 | 0.929 | **0.970** |
| Canadian | 66 / 70 | 0.943 | 0.917 |
| Kama | 62 / 70 | 0.886 | 0.873 |
| **All kernels** | **193 / 210** | **purity 0.919** | **ARI 0.773, NMI 0.728** |

![PCA coloured by true variety](figures/04_pca_true_varieties.png)

When I coloured the PCA plot by the true variety, the picture made sense straight away. The varieties line up along the size axis: Canadian on the left, Rosa on the right, and Kama in the middle, overlapping both.

### Kama, the variety in the middle

The finding I found most interesting is that **every single one of the 17 mistakes involves Kama**. Rosa and Canadian are never confused with each other. Kama and Canadian swap 10 kernels between them, and Kama and Rosa swap 7.

![Clusters vs true varieties with the disagreements circled](figures/08_clusters_vs_truth.png)

The cluster profiles explain why:

![Cluster profiles in z-scores](figures/09_cluster_profiles.png)

| Cluster | Mostly | Typical kernel |
|---|---|---|
| 1 | Rosa | **Large**: about 1.2–1.3 standard deviations above average on every size measurement (area 18.5) |
| 2 | Kama | **Medium-sized and compact**, with low asymmetry (2.7) and a short groove |
| 0 | Canadian | **Small and narrow**: the least compact and the most asymmetric (4.7) |

Kama is the "in-between" kernel: smaller than Rosa, larger and rounder than Canadian. So any Kama kernel at the edge of its range looks like a neighbour.

- **Small or lopsided Kama kernels** were grouped with Canadian.
- **Large Kama kernels** were grouped with Rosa.
- **A few unusually Kama-like Rosa and Canadian kernels** ended up in the Kama cluster.

These are not random errors. For each kernel I measured how much closer its nearest cluster centre is than the second-nearest. The misplaced kernels have a median margin of only 0.41, against 1.94 for the rest. 11 of the 17 sit among the 15% of kernels closest to a boundary. In other words, the algorithm hesitates exactly where a human would.

The cluster centres also land almost exactly on the true variety averages. For area, they are 11.86 against 11.87 for Canadian, 14.44 against 14.33 for Kama, and 18.50 against 18.33 for Rosa.

### Is the structure real, or did K-Means invent it?

K-Means always returns the number of groups you ask for, so I wanted evidence that these three are real:

- **Other algorithms find the same groups.** Ward hierarchical clustering reaches purity 0.929 and ARI 0.797, and agrees with my clusters at ARI 0.836. A Gaussian mixture reaches purity 0.895 and ARI 0.723.
- **The result does not depend on luck.** K-Means gives the *identical* answer from 25 different random starting points.
- **Purity on its own would mislead.** It keeps rising as you cut the data into more clusters and reaches 1.0 when every kernel is its own cluster. ARI, which corrects for chance, peaks at k = 3 and falls to zero:

![Purity, ARI and NMI as k grows](figures/07_purity_vs_k.png)

### How close is this to the best possible?

A perfect score may simply not be reachable with seven measurements, so I wanted a realistic maximum. I trained two supervised models that **are** allowed to see the labels, and scored them on kernels they had not seen (10-fold cross-validation):

| Model | Share of kernels correct |
|---|---|
| Linear discriminant analysis (sees the labels) | **0.962** (±0.036 across folds) |
| Logistic regression (sees the labels) | 0.933 (±0.053) |
| **My K-Means, k = 3 (never sees the labels)** | **0.919** |

Even with the answers, the best model misses about 4% of kernels, because some kernels from different varieties measure almost the same. My clustering, with no labels at all, lands only about 4 percentage points below it. The supervised scores come from held-out kernels while K-Means is scored on all 210, but K-Means never used the labels to fit anything, so I think the comparison is fair.

### The 0.957 I turned down

This was the hardest decision in the project. Five of the seven measurements mostly repeat the same size information, so I tried clustering on **decorrelated ("whitened") principal components** instead. The first three columns use no labels, so they are what I could fairly have used to decide. The last two need the labels:

| Feature space | Silhouette at k = 3 | Best k by silhouette | Lowest ARI over 10 seeds | Purity | ARI vs varieties |
|---|---|---|---|---|---|
| **7 scaled features (my analysis)** | 0.401 | 2 | 1.000 | 0.919 | 0.773 |
| 3 whitened components | 0.326 | 3 | 1.000 | 0.890 | 0.708 |
| 4 whitened components | 0.270 | 3 | 0.931 | **0.957** | 0.876 |
| 5 whitened components | 0.229 | 6 | 0.944 | 0.952 | 0.865 |
| 7 whitened components | 0.164 | 5 | 0.583 | **0.957** | 0.876 |

A purity of 0.957 looked very tempting. I still kept my original analysis, for three reasons:

- **Nothing without labels points to the good versions.** By the label-free columns, the most convincing whitened version is the one with 3 components, and it scores *worse* (0.890). I could only tell that 4 or 7 components work better by looking at the answer key.
- **Some versions are unstable or point to a different k.** With 7 components the answer changes with the random start (ARI as low as 0.583), and the silhouette prefers k = 5.
- **Choosing by purity would break the rule of the project.** If I picked the feature space because it matches the hidden labels better, the higher score would no longer measure what clustering finds on its own. It would measure how well I tuned it to the answers.

To me, this is the real lesson of the project: a better number is only worth having if you could have found it honestly.

## 5. What it means

Without labels, K-Means recovered **most of the real variety structure**. About 92% of kernels landed with their own variety, close to the 96% a supervised model reaches when it is given the labels. The three groups also have a clear physical meaning: Rosa stands out mainly by its size, while Kama and Canadian differ in shape (compactness and asymmetry) as much as in size.

The structure is **real but soft**. The varieties overlap, so about one kernel in twelve ends up in the wrong group, and every one of those involves Kama.

Some limits I want to be clear about:

- **The choice of k was a judgement call.** The elbow said 3 and the silhouette said 2. The purity score only means something together with that choice.
- **The PCA plots are illustrations, not evidence.** They show 89% of the variation, while the clustering itself works in all seven dimensions.
- **All kernels come from one study and one set of fields.** A new harvest or a different mix of varieties could move the boundaries.

**Ethics.** K-Means always produces groups, whether or not they are real. If the co-op treated my clusters as definite varieties, about 8% of kernels would be mislabelled. Because every error involves Kama, decisions based on the clusters would be systematically less reliable for one variety. The claim I am comfortable making is this: *the measurements are consistent with about three varieties, which can be separated for roughly 11 kernels in 12.* That should be confirmed on a labelled sample, and kernels close to a boundary should go for a manual check.

## 6. Reflection

**What worked well**

- **Keeping the labels hidden until the end.** It kept me honest: I fixed k before I saw a single variety, so the final check was a real test.
- **Using more than one diagnostic.** When the elbow and silhouette disagreed, the extra label-free checks turned a guess into a decision I could defend.
- **Looking at the mistakes one by one.** It turned "17 errors" into a clear story about Kama sitting between the other two varieties.

**What was difficult**

- **There is no single correct k.** The standard tools disagree, so I had to weigh evidence instead of reading an answer off one chart.
- **Not over-reading the PCA plot.** It looks convincing, but it hides 11% of the variation and the clustering works in seven dimensions.
- **Turning down a better-looking score.** Seeing purity jump to 0.957 and then deciding not to use it was harder than I expected, but I believe it was the right call.

**What I would improve with more time**

1. **Choose the features fairly.** Decorrelated features can score higher, but nothing without labels says which version to use. I would look for a label-free way to choose, or use a small labelled sample to choose honestly.
2. **Use soft assignments.** A Gaussian mixture gives each kernel a probability for each group, so the co-op could flag uncertain kernels for a manual check instead of forcing every kernel into a group.
3. **Test on new data.** Kernels from other harvests would show whether the three groups and their boundaries hold up outside this sample.
4. **Try model-based ways of choosing k.** For example, BIC for mixture models, as another check alongside the gap statistic.

## 7. How to run it

```bash
pip install -r requirements.txt
jupyter notebook Final_Project_Option3_Seeds_Clustering.ipynb   # then: Run All
```

The notebook runs top to bottom in about 20 seconds. Every random step uses `random_state=42`, so it reproduces every number in this README exactly. I last ran it with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9, SciPy 1.18 and Matplotlib 3.11. If `data/seeds.csv` is not next to the notebook (for example on Google Colab), the notebook downloads the data from OpenML instead.

## 8. Project website

The file `index.html` is a one-page visual summary of the project. It is published with GitHub Pages at **https://marielandreou.github.io/seeds-clustering-portfolio/**, and it links to the full notebook.

## 9. Repository structure

```
seeds-clustering-portfolio/
├── README.md                                      ← this report
├── Final_Project_Option3_Seeds_Clustering.ipynb   ← completed notebook (outputs visible)
├── requirements.txt
├── data/seeds.csv                                 ← UCI Seeds data (7 features + variety code)
├── figures/                                       ← all charts, saved by the notebook
├── report/                                        ← HTML and PDF exports of the notebook
└── index.html                                     ← one-page project website (GitHub Pages)
```

## References

- Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P. A., Łukasik, S., & Żak, S. (2010). Complete gradient clustering algorithm for features analysis of X-ray images. *Information Technologies in Biomedicine*, Springer.
- Seeds data set, UCI Machine Learning Repository. https://doi.org/10.24432/C5H30K
- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society: Series B*, 63(2), 411–423.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
