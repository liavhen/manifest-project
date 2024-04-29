# AdaBoost Classification based on ManiFest Feature Selections
As part of "Mathematical Methods in Data Science" course, this project reviews the work “Few-Sample Feature Selection via Feature Manifold Learning”, by David Cohen, Tal Shnitzer, Yuval Kluger and Ronen Talmon. 
The authors suggest a supervised algorithm to perform feature selection, termed “Manifest”. By using class labels, Manifest finds and scores the most discriminative features of the data, with respect to the different classes. In my project, I suggest examining the quality of the feature selection by constructing a weak-classifier based on each feature, and plugging it to AdaBoost, a learning algorithm that integrates multiple weak classifiers into a stronger classifier with boosted performance. This way, the classification error will be used to evaluate the algorithm. 
In my work, I conduct a comparison between Manifest and another well-known supervised feature-selection algorithm ReliefF, which is considered state-of-the-art in the task of supervised feature selection. The experiments show that Manifest is indeed competitive with ReliefF, even with small number of training samples.

## Results and Conclusions
AdaBoost classification error based on features that were selected and proposed by the following algorithms was measured and examined:
1. Manifest
2. ReliefF
3. Random selection of features

The results exhibit the following dependency of the classification error in the number of selected features, or the number of observed training samples:

<img width="493" alt="image" src="https://github.com/liavhen/manifest-project/assets/94541934/e835af82-edf8-462f-bfce-a407bc7050f2">


<img width="495" alt="image" src="https://github.com/liavhen/manifest-project/assets/94541934/b9658279-083a-42fc-8eda-7b8b644934bc">


<img width="501" alt="image" src="https://github.com/liavhen/manifest-project/assets/94541934/bc11e03b-220f-45c8-af54-0c9bd54e6c5e">


Therefore, the following conclusions were made:
1.	There is no doubt that employing algorithmic feature selection methods significantly enhances performance compared to random selection.
2.	In this task, wrapped by AdaBoost, ReliefF achieves slightly better mean performance overall.
3.	The decline in error for AdaBoost(Manifest) is less rapid with an increase in the number of features compared to AdaBoost(ReliefF). This suggests that Manifest identifies fewer features as discriminative. This deduction is supported by Figure 2, which illustrates that fewer features receive high scores in Manifest compared to ReliefF.
4.	ReliefF appears to exhibit greater stability, as evidenced by its smaller variance in classification error compared to Manifest. This variance arises from the random selection of training samples in each experiment, as well as variations in their sizes across experiments. This deduction is supported by the narrower distributions of each "violin" plot in Figure 5 compared to those in Figure 6 (when taking into account the scale of the vertical axis).

Further details on the experiments, results and implementation details can be found in the code and in the documents under "docs" directory. 

## Miscellaneous

This project marks the end of the Mathematical Methods in Data Science course at Tel-Aviv University's Faculty of Electrical Engineering in 2024. 
The main aim is to take a new look at ManiFeSt, beyond what was discussed in the original paper.

The original paper:
https://ronentalmon.com/wp-content/uploads/2024/01/Cohen_ICML_2023.pdf
