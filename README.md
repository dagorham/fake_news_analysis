# fake_news_analysis
Machine learning classifiers for real and fake news.

The general analysis can be found in the Jupyter Notebook.

The file blender.py contains an implementation of a stacked generalizer model. 
In stacked generalization, multiple "low-level" learners are trained using KFold
cross validation. For each fold, the output probabilities on the hold-out samples are
recorded. These output probabilities then become a new feature space for a high-level
or "meta-learner". Stacked generalization can be implemented using an arbitrary number
of levels.

The blender class is designed to be used with scikit-learn. It takes a dictionary 
of lists of the forms:

{n : [learner_object_1, learner_object_2, ... learner_object_m]}

Where n is the level of the model and the list is a list of scikit-learn objects. If
the entire size of the generalizer is L levels, the L entry in the dictionary has a
value with only one learner, rather than a list. This is the meta-learner.