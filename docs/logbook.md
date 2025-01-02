## 2024-11-04

The MCTS agent is not able to beat the before agent when using 500 simulations. Furthermore, the MCTS current implementation takes 8h to run on the test set. We need to check two things:

1. Is the MCTS agent implementation correct?
2. Can we speed up the MCTS agent?

To address 1., we will need to create basic tests where the selection is obvious. For instance, if A happens before B, the agent first action should be `Relation(end A < start B)` which would make all the other relations inferable. We should also implement extra logging to stores every action that has been taken. From that we can create a dashboard to visualize the actions taken at each step. Something like the MuZero dashboard.

<img src="assets/muzero_dash.png" alt="MuZero Dashboard" height="400">

Regarding 2., after doing some profiling one can conclude that most of the time is spend on computing the temporal closure of the the timeline, and therefore, not a problem with the MCTS agent itself. However, improvements can still be made by using arrays rather than the Node class.

## 2024-11-07

To simplify the task, a new dataset, [Temporal Games](https://huggingface.co/datasets/hugosousa/TemporalGames), was build. This dataset has some partitions, "two", "three", ..., that represent the number of entities per entry in the corpus. That is, all the documents in the "two" partition have two entities. Important to note that all the entries have at least one temporal link that was manually annotated. This data should be better to identify what are the relation types that the classification models struggle more to label.

## 2024-11-08

To train the Temporal Game agent it would be useful to first build a classification model that matches human agreement or is close to it. This model should be useful to answer to a few questions. 

1. What is the relation type that is harder to predict.
2. what is the endpoint pair that is harder to classify? Is it start-start, start-end, end-start or end-end?

To access this, one can train classification models that focus in just one of the questions at a time. For instance, to find what is the relation type that is harder to predict we can train a model on a corpus only has two labels: 1 for the relations we want to estimate the complexity of the classification; 0 if the relation is not that. This will also help to identify what are the relations that the model confuses the most.

## 2024-11-12

Set the validation dataset to also be augmented when the training dataset is augmented. This is because the training was stopping to early due to early stopping. By balancing the valid dataset the training process is expected to last longer as the validation loss will be more stable.

Increasing the batch size improved the generalization of the model that trains without augmentation or balancing.

Cosine annealing also seems to help on the training process. Initially it was set to only decrease once, with the number of steps set to the total number of training steps, however setting it to decrease in one epoch and increase in the following one seems to be better.

Considered using MATRES as part of the evaluation. However, MATRES results from the annotation of documents from TimeBank dense, which in itself, resulted from the annotation of documents from TimeBank, which we use in the training process. Therefore, we will only add TimeSET for our evaluation.

When adding the timeset dataset we found that the number of unique texts in the dataset is 4044. With 1671 of the relation BEFORE, 1671 of the relation AFTER, and 702 of the relation COEX. Upon closer inspection, we found that the after entries are in fact before relations that were inverted. Furthermore, there seems to be some duplicates as if one checks what are the unique texts + arg1 + arg2 in the 1671 AFTER entries there are 1663 unique texts (the same is true for the BEFORE entries). The COEX entries are unique.

Forgot to drop the relations between the start and the end of the same entity of the q timelines dataset (which is always "before"). This has been inflating the effectiveness of the model. This has been fixed.

On another note, the q timelines context now only has tagged the entities being classified. This should increase the quality of the data and improve the learning process.

## 2024-11-13

Now the learning rate decreases after each epoch restart. That is, we are using a cosine annealing scheduler with restarts that at the end of each epoch reduces the learning rate by 5%.

Why does the validation loss increases while the validation accuracy is stable?

Looking at the image bellow one can see that the model is still learning from the training data. However, the validation loss is increasing. Furthermore, the validation accuracy is stable.

<img src="assets/wandb_valid_loss.png" alt="Loss and Accuracy" height="400">

Another interesting remark is the fact that the accuracy for the None label is 0% throughout the all training procedure. It could be that the model is not able to learn the label, or that can be that the data has low quality for this label.

Found some bad entries in the data. Samples from like "Putin <5>faced</5> whistleswhistleswhistles and boosboosboos" which seem to come from GraphEve as that is the only dataset for which the entities is just a number. This was due ot the fact that GrahEve had several ids to the same entity. This has been fixed.


## 2024-11-14

None of the generated synthetic answers contain the expected tags and the expected text. We need to improve the prompt generation process.
