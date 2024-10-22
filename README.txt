Preface:

As a general overview, I was tasked with gathering roughly 10000 documents and classifying them via sentence embeddings. To this end, I utilized the Reuters-21578 dataset. Reuters-21578 is well suited for multi-label classification given the extensive annotation of documents with multiple topics/labels. It is an ideal candidate for evaluating effectiveness of sentence embeddings in these sorts of classification tasks. As for the actual models used to classify the documents, I saw fit to use first BERT, and then S-BERT. I chose these models specifically because they have shown strong performance in classification tasks such as this take-home. Although S-BERT is oriented specifically towards producing sentence embeddings, I chose to first experiment with its precursor, BERT, because I wanted to approach this topic from the 'ground-up' so-to-speak. Experiments utilizing BERT have titles concluding with 'BERT', and experiments utilizing S-BERT have titles concluding with 'SBERT'. Experiments ending in 'lite' indicate that, instead of fine tuning the model in an end-to-end process, I've instead just added a classifier on top of the model and trained the classifier.

As a warning, I have submitted the take-home on document/text classification without having fully completed all the plans I have laid out. As it is, the take-home covers experiment a-1BERT (initially described in note #1, some changes to the cleaning/splitting pipeline described in note #2), experiment a-1-1BERT, a-1-1SBERT, a-1-1SBERT-lite (change to splitting pipeline described in note #3), as well as the very beginning of experiment b-1BERT.

Experiment a-1BERT covers my initial data extraction + cleaning process, alongside my attempts at filtering classes and balancing class distribution for the training/testing as well as training/validation/testing sets (further details found in notes below). a-1-1BERT is a version of a-1BERT shortened to omit experimentation relating to class balancing and omit experimentation regarding inclusion of a validation set. Instead, experiment a-1-1BERT only has a train/test split and balances classes immediately via thresholding topics based on a lower limit of associated documents. Experiments a-1-1SBERT and a-1-1SBERT-lite follow this convention as well.

Experiment a-1SBERT-lite is similar to a-1-1BERT, but utilizes the S-BERT model instead. I also included a notebook called a-1SBERT-lite, which, instead of fine tuning the S-BERT model with end to end training, just trained a separate classifier on top of the fixed embeddings.

Experiment b-1BERT covers my process on fine-tuning BERT for the ModApte split of the Reuters-21578 dataset. The cleaning pipeline is naturally far more diminished in this version given that we are trying to adhere as strictly as possible to the ModApte split. In light of this, there is also no attempt at creating a validation set. This time, instead of extracting relevant data from the .SGM files associated with the Reuters-21578 dataset, I am pulling the dataset and the associated ModApte split directly from huggingface's datasets library. I am also going to go ahead and use the trainerAPI from huggingface to accomplish training and evaluation as opposed to doing it manually like in the type 'a' experiments.

Experiment b-1SBERT will be similar to b-1BERT, but will utilize the S-BERT model instead.

It should be noted that, due to time constraints, I have not completed as much hyperparameter tuning as I wanted to. These will be completed and sent as part 2 of this submission if you are open to it.

Further Plans:

Alongside completing the rest of the planned experiments, as well as further fine tuning the experiments I've already compiled, I would like to test out other SBERT models such as 'all-roberta-large-v1' as well as 'stsb-roberta-large', likely through renting cloud-GPU resources.





Notes/Planning:

Note #1. Some documents that do not have topics/labels should have had topics/labels (but do not due to errors or omissions during indexing process), and others don't have topics/labels simply because they were uncategorizable, thus making them true negatives for all categories. Because of this ambiguity, it's not safe to assume that all documents without topics/labels are true negatives, and it's not safe to assume that they should all be excluded from analysis. Documents that should have had topics/labels, regardless of whether they do or not, are marked with 'TOPIC = YES'. 

Thus, we'll try a couple different experiments:

Type 'a' experiments will be according to our own data cleaning pipeline. Each type 'a' experiment will have newline characters removed from relevant input text, additional spaces removed from relevant input text, and HTML/SGML entities removed from relevant input text. In addition, empty and/or nonexistent title+body combinations (input text) will be removed. Moreover, all included documents must be marked with 'TOPIC = YES', so that even if they are lacking an explicit topic assignment, they still should have had one. We'll also have to make sure the training and testing set both have at least one document associated with each label/topic we keep in our dataset. That means we'll have to remove from the label/topic lists in the appropriate column those labels/topics with only a single associated document (note, this is not the same as removing said documents themselves given that each document can be associated with multiple labels).

a-1BERT/a-1SBERT. First, we'll try training without including any of the unlabeled documents in either our training-set, or our testing set.

a-2BERT/a-2SBERT. Next, we'll try training, this time including the unlabeled documents in our training set, but not in the testing set.

a-3BERT/a-3SBERT. Next, we'll try training without including any of the unlabeled documents in our training-set, but including them in our testing set.

a-4BERT/a-4SBERT. Next, we'll try training, this time including the unlabeled documents in our training set, as well as the testing set.

a-5BERT/a-5SBERT. Unlabeled data is marked as a separate topic 'unlabeled, and included in both training and testing sets.



Type 'b' experiments will refrain from using a custom cleaning pipeline due to the recommendations in the notes accompanying the Reuters-21578 dataset (modapte is widely used for classification tasks regarding the topics/labels for this dataset, comparison of results is therefore far more readily available specifically when using the modapte test set). These experiments will make use of the recommended ModApte test split described in the dataset's notes, although we may make adjustments to the training set, specifically with regard to whether or not those labels/topics with single documents are included or not, as well as with regard to whether or not non labeled data is included in the dataset. Seeing as huggingface has the dataset available via a specific library, we will try using that instead. In addition, I see that there is a training API available from huggingface as well. I will try to utilize that instead of the manual approach for the type 'a' experiments.

b-1BERT/b-1SBERT. Training set excludes non labeled data.

b-2BERT/b-2SBERT. Training set includes non labeled data

We may also want to consider running a separate set of experiments that utilize only the body.



Note #2. I probably should have foreseen this earlier, but it would seem that ensuring that both the training and the testing sets have at least one document of each label/topic that remains in the dataset following removal of those labels/topics with less than 2 documents associated is not possible. This is because certain rare topics/labels may share documents among them. Consider the following simplified example:

Document 1: Topic A, Topic B
Document 2: Topic A, Topic C
Document 3: Topic B, Topic C

that is,

Topic A has 2 documents: Document 1 and Document 2.
Topic B has 2 documents: Document 1 and Document 3.
Topic C has 2 documents: Document 2 and Document 3.

In this case, if we attempt to split the data such that both training and testing sets have at least one document per topic, it's not really possible.

Given this, there are a couple ways we can approach the problem:

a. We could either duplicate overlapping documents such that both the training and the testing set include at least one document for each topic. However, this is not ideal as this causes data leakage.

b. We could just leave certain rare topics/labels exclusively in the training set (testing set doesn't make sense considering we couldn't accurately classify those documents with topics/labels exclusively in the testing set without having seen examples of said topics/labels in training first).

c. We could augment the data for rare topics/labels.

d. We could use cross validation.

e. We could simply remove certain rare topics/labels until we can achieve a split such that both training and testing sets have at least one document associated with each label.

f. We could group rare topics/labels into a single category.

Given time constraints, I have decided to go forward with two seperate methods:

a. Collate info on which rare topics/labels appear in testing but not training, and training but not testing. Try and get a balanced spread where possible, and where not possible, relegate topics/labels to training. Doesn't make sense to keep them in testing if we're not training on them.

b. Remove those topics/labels that cannot be split evenly altogether. Shouldn't be too much of a problem considering they have very few associated documents to begin with.

I will proceed first with method 'b' on how to handle even splitting of remaining topics/labels as this makes best use of the remaining time I have for this take-home. => FOLLOW UP => After some testing, it would seem that removing those rare topics/labels that could not be evenly split still does not guarantee a train/test split such that each remaining topic/label has at least one occurence in both. Given this, I have decided to proceed with a method where we cutoff those labels/topics that have less than 5 documents associated with them. This should guarantee that our train/test split has at least one document associated with each of the labels/topics that remain. Given that, for the 'b' type experiments, we're adhering as much to the modapte split as possible, we will not be utilizing a cutoff of 5 either, instead opting for a cutoff of only 2.

Note #3. Initial training results for experiment a-1BERT show a better f-1 score for a train/test split of 80:20 as opposed to a train/validation/test split of 60:20:20. I figured the validation set would be a good addition given that it helps us tune hyperparamaters without overfitting via early stopping and checking for generalization, and given that it can help us monitor performance while training. However, based on the evaluation metrics for train/test vs train/validation/test in experiment a-1BERT, I have decided not to utilize a validation set in experiment a-1-1BERT. Experiment a-1-1BERT will focus on cleaning up experiment a-1BERT, specifically sections pertaining to unused data cleaning, sections where we were testing removal of topics based on how well they could be split evenly, the section where we initiate a validation set, and the section regarding training and evaluation for when we have a validation set. Generally, the additional '-1' at the end will denote those experiments that opt not to use a validation set.

