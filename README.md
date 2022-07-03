# Using Contextual Multi-Armed Bandit for Diabetes Medication Recommendations


## Abstract
Diabetes is a widespread chronic disease which can have many treatment options complicated by co-morbid conditions. We use the reinforcement learning algorithm contextual multi-armed bandit (MAB) in order to create a medication recommendation system with the goal of supporting clinical decision making. We examine and tune four action-selection methods the for the MAB and find the bootstrapped upper confidence bound gives the best experimental results.

## 1. Introduction
Diabetes is a chronic disease which occurs when the body either cannot produce the hormone insulin itself or is unable to effectively use the insulin it produces (WHO, 2016). It is an increasingly prevalent disease, estimated to affect over 10.5% of the world’s adult population and result in related health care costs of 966 billion USD in 2021 (Sun et al., 2022). Diabetes treatment often includes a medication com- ponent, however, treatment plans are often complicated by the complex interactions between diabetes and its medica- tion with other co-morbid conditions (?). This motivates the need for solutions assisting medical professionals in efficiently and effectively devising patient treatment and medication plans which are suited to a patient’s particular needs.

As the availability of healthcare data expands, there has been a growing research interest in applying well-known machine learning techniques to clinical health care problems. In particular, reinforcement learning (RL) has been studied for its use to support clinical decision making regarding diagnosis, outcome forecasting, and treatment (Liu et al., 2020). Traditional clinical decision making is a sequen- tial decision-making process based on procedure guidelines and best practices which are not personalized to the patient and do not always take co-morbid conditions into account (Gottesman et al., 2018). These decisions can be reformu- lated as a recommendation problem, where the goal is to recommend the best option for the given input. In the case of diabetes treatments, the problem for the agent becomes to recommend the optimal treatment for each patient. Rein- forcement learning is a well explored solution for general recommendation problems such as recommending news articles or movies to a given user based on their past preferences and the preferences of other similar users. RL is particu- larly suited to clinical problems as it does not necessarily require a full formulation of the mathematical model of the environment which would be difficult to create, and it has the notion of policy which meaningfully corresponds to a clinical treatment policy.

Nonetheless, there are a number of challenges in applying RL to healthcare problems which do not naturally exist in other domains. Unlike scenarios where an RL agent learns to play a game or recommends products, in health care set- tings learning must be done only through observing data rather than running numerous experiments due to the uneth- ical nature of using patients to train the agent. Moreover, this can often result in having no data on certain state-action pairs which can skew the agent’s policy, especially in the case where the known actions often have adverse outcomes (Gottesman et al., 2018).
The goal of this study is to explore the use of reinforcement learning as a method of recommending the appropriate treat- ment for each patient. We train agents using a variety of contextual multi-armed bandit (MAB) algorithms on a data set composed of almost a decade of United States hospital admissions for people with diabetes and the treatments they received (Strack et al., 2014). Using this data, we are able to train the agents to select appropriate treatment recommenda- tions and compare which contextual MAB algorithm is most effective for this problem. Since insulin is the most common drug used to treat diabetes and the dosage of insulin can have a large impact on diabetic patients (WHO, 2016), we focus our medication treatment recommendations on how best to change the patient’s insulin dosage.

The remaining sections of the paper are organized as follows: Section 2 examines related work, section 3 gives a more exact formulation of the problem, section 4 details our approach including the data set detail, the necessary data processing, and the specific methods used. Experimental setup and results are discussed in section 5 and finally, there is a discussion and conclusion summary in sections 6 and 7 respectively.

## 2. Related Work
There has been extensive work done using MAB, and specif- ically contextual MAB for recommendation systems. Re- cently a comprehensive survey was completed detailing state-of-the-art in MAB for recommendation systems and covers over 1300 works published over the last 20 years, which exemplifies how synonymous MAB and recommen- dation systems have become (Silva et al., 2022).

In contrast, previous work on diabetes treatment recom- mendations do not use MAB algorithms, instead opting for formulating a full Markov Decision Process (MDP) or using a deep Q-network (DQN).

The 2013 paper (Asoh et al., 2013) appears to one of the first works on RL for diabetic treatment recommendations. They create a relatively simple MDP for the problem where pa- tient state is based solely on the patient’s level of HbA1c and actions are a subset of common medication combinations. The resulting rewards are based on how close the selected action is to the actual doctor prescription. To evaluate the model a 90:10 training/testing split as well as calculating the root-mean-squared error between cumulative rewards and estimated state values.

The paper of (Sun et al., 2022) specifically tackles treatment recommendations for type 2 diabetes. It uses a similar 80:20 training/testing split as in our paper, but uses DQN instead of contextual MAB. Due to wider availability of data (in terms of more tracked features for each patient) this study is also able to estimate both short and long term effects of actions while we are limited to only overall optimality of actions. However, to evaluate the model they use a very similar accuracy measure, comparing the model actions with physician actions. Their results show model actions leading to a potential decrease in patient fatality but only has an accuracy of 43%.

The recent paper of (Oh et al., 2021) creates a MDP-based treatment recommendation system by fully specifying the states, actions, reward functions, and transition probabil- ity matrices. Again this study benefits from a robust data set as a foundation which is not publicly available. The states are engineered by creating formulas for the severity of the patient and the patient’s risk factors, using methods not dissimilar from the feature engineering we use to reduce the our data dimensions. The action space is also based on recommending either single, double, or triple therapy (the number of medications the patient should receive) rather than the insulin based recommendations we use. To vali- dated their results, they are compared with doctors’ real life prescriptions which showed an accuracy of 68% and 61% for males and females respectively.

Lastly, there has been previous work done using the same di- abetes data set which we make use of but for predicting early hospital readmission rather than medication recommenda- tions. One such example is (Sarthak et al., 2020) which is able to effectively use the data set for this readmission problem and uses similar data processing methods.

## 3. Problem Formulation

Contextual MAB is a RL algorithm based on the idea that at each time step the agent is given the context for the problem and then based on the current policy the agent chooses which action/arm to choose and receives the corresponding reward which it then uses to update its estimates for what the best action is for that given context. For our problem, the context given to the agent is the patient data such as their age, diagnoses, lab results, etc. and then the agent can choose which treatment to use as its action. Based on whether the agent chose an optimal action or not, the agent will receive a reward of either 1 or 0 respectively. This is based on the assumption that physicians’ actual decisions and prescriptions from the dataset are optimal. In our formulation the possible actions are to choose no insulin prescription, no change from current insulin dosage, increase insulin dosage, or decrease insulin dosage. This formulation mirrors a physicians decision making process for adjusting the insulin dosage of a newly admitted diabetic hospital patient.

Formally, we describe the problem based on the notation from (Li et al., 2010). A contextual-bandit algorithm A proceeds in discrete trials t = 1,2,3,.... In trial t:

1. The algorithm observes the context xt, which contains the current patient data pt and the set At of actions.
2. Based on previous observed rewards and the action selection strategy, A chooses an action a ∈ At and re- ceives the corresponding reward rt = 0 or 1 depending on if a is optimal for user pt or not.
3. Action selection strategy is improved using the new observation, (xt, at, rt)

## 4. Approach
Our approach to this problem first involves processing the data set to standardize the categorical variables and reduce the dimensions of the data using feature engineering. The data is then split into training and testing sets in an 80:20 split. Four different contextual MAB algorithms are used to train the agents in the experiments: Explore-First; Epsilon- Greedy; LinUCB; and BootstrapUCB. Different parameters for the algorithms are experimented on using the processed training data. The learning curves of the algorithms are plotted for comparison, and the accuracy of each is assessed using the test set and compared to a typical logistic model for multi-classification of insulin dosage information.

### 4.1. Data Processing
The data set we use is the publicly available ”Diabetes 130- US hospitals for years 1999-2008 Data Set” from the UCI Machine Learning Repository (Strack et al., 2014). Pub- lished in 2014, it contains 101,766 instances of clinical record data of diabetic patient encounters captured, and it has 55 numeric or nominal data attributes. Some of the key attributes that are important to train the RL model are the medication data which is the dosage information of insulin; demographic data such as age and race; lab results such as glucose serum and A1C level; and admission, discharge, and transfer (ADT) data such as time in hospital.

Any duplicates of records are removed. Since the same patient could have multiple encounters, this could intro- duce bias in the data instances and only the first encounter is kept. The attributes ‘weight’, ‘payer code’, and ‘medi- cal specialty’ are removed due to high percentages of miss- ing values (97%, 40%, and 49% respectively). The identifier attribute ‘patient nbr’ is also removed. Missing values are imputed by replacing them with the mode values. For the at- tributes on diagnosis and ADT, their values are grouped into the general categories. The dosage information of insulin is kept.
Feature engineering is also conducted resulting in two more attributes, ‘health index’ which combines the fea- tures ‘number emergency’, ‘number inpatient’, and ‘num- ber outpatient’, and ‘severity of disease’ which combines ‘time in hospital’, ‘num procedures’, ‘num medications’, ‘num lab procedures’, and ‘number diagnoses’.

Outliers are then removed by calculating the Z scores for the numeric variables ‘age’, ‘health index’, and ‘sever- ity of disease’ when the Z score is less than 3. The nominal values are then one-hot encoded (to be suitable for logistic regression which is responsible for learning the context of input data in some MAB algorithms). Due to the high di- mension resulted from this, feature selection is conducted by using the Chi-squared test with a 95% confidence interval to remove attributes that are not highly dependent on the insulin data.

Lastly, the processed data set is split into training and test set with a ratio of 80:20. When needed in experiments, the data
set is also under-sampled to ensure a balanced distribution of the insulin dosage information. Over-sampling is not used as it likely to lead to over-fitting by duplicating existing records, or is impractical for high dimensional data by synthesizing examples.

### 4.2. Contextual Multi-Armed Bandit Algorithms
Depending on the problem, different methods of selecting the agent’s next action produce the best results. For our purposes we explore four different action selection methods which are commonly used in recommendation problems or have been shown to produce good results (Silva et al., 2022), (Cortes, 2019). A summary of each with their required hyper parameters is given below.

### 4.2.1. EXPLORE-FIRST
Explore-first is a relatively naive method, often used as a baseline for method comparison (Cortes, 2019). It consists of a set number of exploration rounds in which the agent chooses an action at random, followed by exploitation in all other rounds in which the agent chooses the maximum valued (greedy) action. The method takes the number of exploration rounds as a hyper parameter.

### 4.2.2. EPSILON-GREEDY
Epsilon greedy is a simple but often effective method which at each step chooses either a random action or greedy ac- tion based on the hyper parameter epsilon value. A random action is chosen with probability epsilon and the greedy action taken otherwise. Alpha (α) is also taken as a hy- per parameter representing the learning rate of the method. Higher alpha values mean the agent will learn faster but could have more variance in the mean reward over time, while lower alpha values represent a slower but more stable rate of learning.

### 4.2.3. LINUCB
Upper confidence bound (UCB) methods are based on the idea that rather than simply choosing the maximum greedy action or a random action, it makes sense to weigh an ac- tion’s probability of being chosen by how close to greedy it is and how often it has been chosen before (which informs how confident the agent is in the correctness of its estimate of the action’s true value). These factors are combined to adjust the agent’s perceived action values such that the ’greedy’ action at a given time step may not be the action with the highest mean reward, but rather one with a decent mean reward that has not been chosen as often.

LinUCB is a way to adjust the idea of UCB which was created for non contextual MAB to use in the contextual setting which needs to efficiently account for more information in its action estimations (Li et al., 2010). As with the epsilon-greedy method, linUCB takes alpha as a hyper parameter.

### 4.2.4. BOOTSTRAPPED UCB
Another variant of UCB, Bootstrapped UCB incorporates the statistical idea of a multiplier bootstrap into the action estimations which can provide a tighter confidence bound, potentially allowing for more confident exploitation, for some problems (Hao et al., 2019). It takes two hyper param- eters, the percentile of reward r estimates and the number of resamples, although the number of resamples is generally chosen based on available computational resource rather than tuning. The percentile of reward r estimates represents what quantile of samples to use as the UCB for each arm in order to choose an arm given a patient context (Cortes, 2019). As the agent learns, the bound should get closer to the expected reward for each arm.

## 5. Experiments
We run a series of experiments on the processed data set, evaluating the mean reward and accuracy of each action selection method with multiple hyper parameter values.

### 5.1. Experiment Setup
Our implementation is achieved by using logistic regression as a base algorithm to improve the agent’s understanding of patient contexts/states since otherwise if the contexts are used as originally given, there are simply too many unique contexts that it is very difficult for the agent to learn []. More specifically, there are multiple independent logistic regression models constructed; for example, there can be one for each arm. They fit to the given training contexts as the input data and the corresponding rewards from pulling an arm as the prediction labels and get updated. They are then responsible for receiving new contexts to predict rewards which are compared to select the best arm with the highest reward. They then get updated again and keep repeating this process, and the interval of updates can be a parameter. In the extreme case, the logistic regression models could update after each step but this takes a substantial amount of run time; in our experiment, it is set to 600 steps for the unbalanced dataset, and 200 steps for the balanced dataset. This interval is often treated as a batch or a round (Cortes, 2019).

The algorithms are run on the unbalanced and balanced data set in hope to eliminate any bias that could be introduced by the unbalanced distribution of insulin dosage informa- tion. Different parameters values are selected for the four algorithms, as summarized in Table 1.

<img width="433" alt="image" src="https://user-images.githubusercontent.com/47286892/177057203-8772c656-6ff7-4218-9f0e-2b29c60a8d07.png">

Table 1. Summary of experiment methods and hyper parameters.

A typical logistic regression model is also constructed as a multi-classification problem predicting insulin dosages as a baseline. (This is not the same as the base model for the contextual MAB algorithms).

### 5.2. Results
Results of Epsilon-Greedy algorithms with an α value of not 0.1 are omitted from Figures 1, 2, and Table 2 because their learning curves and test accuracy results are too close to each other. Only those with an α of 0.1 are kept.

<img width="430" alt="image" src="https://user-images.githubusercontent.com/47286892/177057208-ae5ae280-28c1-4d37-826a-f0db7987c463.png">

Figure 1. Learning curves of algorithms for unbalanced data.

<img width="430" alt="image" src="https://user-images.githubusercontent.com/47286892/177057226-70897a7e-aa15-42f0-8ac1-198587ca1d1c.png">

Figure 2. Learning curves of algorithms for balanced data

<img width="402" alt="image" src="https://user-images.githubusercontent.com/47286892/177058964-dfe66d44-29d8-4041-9c38-64f87d9d86c0.png">

Table 2. Summary of the best test accuracy.

## 6. Discussion
Despite being a common method, LinUCB did not do very well in practice. In fact, it suffers more when the amount of data decreases as shown by the low test accuracy and the decreasing learning curve for the smaller balanced dataset. This is likely due to LinUCB being intended for continuous rewards (while we use binary rewards) (Cortes, 2019).

The dataset has a few limitations as it contains sparse nom- inal variables which need to be transformed into one-hot encodings, possibly resulting in overfitting, inaccurate fea- ture importances, or high variance. Moreover, when the dataset is under-sampled to achieve a balanced distribution over the insulin dosage information, the amount of data decreases substantially losing its information and statistical power.

Regarding the Epsilon-Greedy algorithm, a higher ε, mean- ing that more exploration than exploitation, results in better performance. This could be due to the high dimension of the data set resulting in a large amount of possible pa- tient states/contexts, making exploitation less of an effective learning technique as compared to exploration since it is more frequent to see new contexts instead of learning about existing contexts.

For the best performing RL model Bootstrapped UCB, it is apparent that a 50% percentile is the ideal value for this hyper parameter, meaning that the 50% percentile of the estimated rewards gives a tight upper confidence bound. As for the other parameter, number of resamples, a smaller number of 5 instead of 10 results in better performance for the balanced data set. This is likely because the number of resamples out of the batch/round size becomes more substantial, resulting in more accurate estimates during the statistical bootstrapping process.

In general, recommendation problems can be converted to a multi-classification problem with the actions being the la- bels, and the contexts being the training examples. However, supervised learning like logistic regression only performs accurately based on the distribution of the training data. For example, if the training data only contains patients having no insulin, then the supervised model does not have any statistical evidence for or against recommending any other dosage information, resulting in poor performance although practically speaking, such an extreme example rarely exists. With the possibility to explore and find a balance between exploration and exploitation, RL models have an advantage in obtaining more accurate estimates of which dosage to recommend compared to supervised learning.

## 7. Conclusion and Future Research
Multiple contextual MAB algorithms are implemented, trained, and evaluated successfully. As compared to a typ- ical machine learning model like logistic regression, RL models in general has the advantage of not being dependent on the distribution of the training data, and Bootstrapped UCB has achieved a higher accuracy when the amount of data is sufficient. LinUCB suffers from non-continous rewards but the rest of the RL models perform closely to each other, with Bootstrapped UCB having slight better results. It is suspected that with more more quality data, the learning of the RL agents could be improved. In the future, other ways of determining patient contexts/states could be exper- imented on, and the reward function could involve more conditions such as the the clinical outcome of patients.

## References

Asoh, H., Shiro, M., Akaho, S., Kamishima, T., Hasida, K., Aramaki, E., and Kohro, T. Modeling medical records of diabetes using markov decision processes. pp. 6, 2013.

Cortes, D. Adapting multi-armed bandits policies to con- textual bandits scenarios. arXiv:1811.04383 [cs, stat], Nov 2019. URL http://arxiv.org/abs/1811. 04383. arXiv: 1811.04383.

Gottesman, O., Johansson, F., Meier, J., Dent, J., Lee, D., Srinivasan, S., Zhang, L., Ding, Y., Wihl, D., Peng, X., Yao, J., Lage, I., Mosch, C., Lehman, L.-w. H., Ko- morowski, M., Komorowski, M., Faisal, A., Celi, L. A., Sontag, D., and Doshi-Velez, F. Evaluating reinforce- ment learning algorithms in observational health set- tings. arXiv:1805.12298 [cs, stat], May 2018. URL http://arxiv.org/abs/1805.12298. arXiv: 1805.12298.

Feb 2020. URL http://arxiv.org/abs/2002. 11215. arXiv: 2002.11215.

Silva, N., Werneck, H., Silva, T., Pereira, A. C. M., and Rocha, L. Multi-armed bandits in recommendation sys- tems: A survey of the state-of-the-art and future direc- tions. Expert Systems with Applications, 197:116669, Jul 2022. ISSN 0957-4174. doi: 10.1016/j.eswa.2022. 116669.
Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ven- tura, S., Cios, K. J., and Clore, J. N. Impact of hba1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. BioMed Re- search International, 2014:e781670, Apr 2014. ISSN 2314-6133. doi: 10.1155/2014/781670.

Sun, H., Saeedi, P., Karuranga, S., Pinkepank, M., Ogurtsova, K., Duncan, B. B., Stein, C., Basit, A., Chan, J. C. N., Mbanya, J. C., Pavkov, M. E., Ramachan- daran, A., Wild, S. H., James, S., Herman, W. H., Zhang, P., Bommer, C., Kuo, S., Boyko, E. J., and Magliano, D. J. Idf diabetes atlas: Global, regional and country- level diabetes prevalence estimates for 2021 and projec- tions for 2045. Diabetes Research and Clinical Prac- tice, 183:109119, Jan 2022. ISSN 1872-8227. doi: 10.1016/j.diabres.2021.109119.

WHO. Global report on diabetes. World Health Organiza- tion, 2016. ISBN 978-92-4-156525-7. URL https:// apps.who.int/iris/handle/10665/204871.
Hao, B., Abbasi Yadkori, Y., Wen, Z., and Cheng, G. Boot- strapping upper confidence bound. In Advances in Neural Information Processing Systems, volume 32. Curran As- sociates, Inc., 2019. URL https://proceedings. neurips.cc/paper/2019/hash/ 412758d043dd247bddea07c7ec558c31-Abstract. html.

Li, L., Chu, W., Langford, J., and Schapire, R. E. A contextual-bandit approach to personalized news article recommendation. Proceedings of the 19th international conference on World wide web - WWW ’10, pp. 661, 2010. doi: 10.1145/1772690.1772758. arXiv: 1003.0146.

Liu, S., See, K. C., Ngiam, K. Y., Celi, L. A., Sun, X., and Feng, M. Reinforcement learning for clinical decision support in critical care: Comprehensive review. Journal of Medical Internet Research, 22(7):e18477, Jul 2020. doi: 10.2196/18477.

Oh, S.-H., Lee, S. J., Noh, J., and Mo, J. Optimal treatment recommendations for diabetes patients using the markov decision process along with the south korean electronic health records. Scientific Reports, 11(11):6920, Mar 2021. ISSN 2045-2322. doi: 10.1038/s41598-021-86419-4.

Sarthak, Shukla, S., and Tripathi, S. P. Embpred30: As- sessing 30-days readmission for diabetic patients using categorical embeddings. arXiv:2002.11215 [cs, stat],
