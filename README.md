# NLP-Movie_Review
Use sklearn to train/test two different models for sentiment analysis of movie reviews.

# Usage:
1. Clone the repository
2. Download the IMDB movie review database [here](https://ai.stanford.edu/~amaas/data/sentiment/)
3. Open your command line and navigate to the directory you cloned in step 1.
4. In your command line, run the __movie.py__ file using the following command:
````
python movie.py trainDir outputFileNB outputFileLSV
````
This command will run the python file, using __trainDir__ as the location of all your training data folders (should be labeled __pos__ and __neg__, __etc__), and will output the information for the Naive Bayes and Linear Support Vector classifiers modeled in this project into __outputFileNB__ and __outputFileLSV__ respectively. See [exampleOutputs](movie.py) for my results.
