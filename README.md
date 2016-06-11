# amazon-sentiment-ml


The dataset for this project was obtained from the SNAP group <a href="http://snap.stanford.edu/data/web-Amazon.html">here</a>. They have about 35M Amazon customer reviews; I have only included the reviews for the Automotive section. The data has the following format:

```
product/productId: B00006HAXW
product/title: Rock Rhythm & Doo Wop: Greatest Early Rock
product/price: unknown
review/userId: A1RSDE90N6RSZF
review/profileName: Joseph M. Kotow
review/helpfulness: 9/9
review/score: 5.0
review/time: 1042502400
review/summary: Pittsburgh - Home of the OLDIES
review/text: I have all of the doo wop DVD's and this one is as good or better than the
1st ones. Remember once these performers are gone, we'll never get to see them again.
Rhino did an excellent job and if you like or love doo wop and Rock n Roll you'll LOVE
this DVD !!
```



### noramlize_reviews.py

Use this script by running `python normalize.py Automotive` in the command line. If you have the other categories, you can put them in instead of Automotive. Normalize.py splits the reviews into three sets of reviews: build, train, and test. This script creates three `.txt.gz` files for the three sets. The build set is not used here; it was used to build the Amazon sentiment lexicon. The train set is used to train machine learning algorithms, and the test set is used to test the machine learning algorithms.  `Normalize_review.py` also only takes reviews that are 140 characters or less because the end goal of this project was to create algorithms to classify Tweets as having positive sentiment or negative sentiment, and Tweets are restricted to 140 characters. This restriction also goes some way to solving another issue: if a review is very long, there could be many subjects discussed, not only the product being reviewed. This makes it a bit harder to say that for sure the entire review is positive or negative, because we really would like to only look at the parts of the reivew that are talking about the product. One way to make this better would be to use some kind of topic identification or named entity identification to pick out the parts of the review that are talking about the product in question, and only use those parts to train the classifier.


### training_raw_algos.py

Use this script by running `python training_raw_algos.py Automotive` in the command line. This script looks through all of the reviews in `Automotive_train.txt.gz` (which is created by normalize_reviews.py), and finds the top 10% of the most common words; these will be used when creating the features to train the machine learning algos. For each document (review), the features will be a dictionary with keys the 10% of most common words, and values either True or False; True if that word appears in the document, and a False if not. The classes used for training these algorithms are "pos" and "neg". The class "pos" is assigned to a document if the customer review was either 4 or 5, and "neg" was assigned to a document if the customer review was 1, 2, or 3. Python's NLTK (natural language toolkit) package was used to train these algorithms.


### sklearn_classifiers.py

This is a helper script that defines a class to make the output of the NLTK classifiers amenable to some of sklearn's functions


### test_algos.py

Use this script by running `python test_algos.py Automotive` in the command line. This script tests the machine learning algorithms on the test set built earlier. This will output accuracies of the machine learning algorithms. One good thing would be to look at recall.
