This is the repository for the COLING 2022 paper: "QSTS : A Question-Sensitive Text Similarity Measure
for Question Generation".

The model for question-class prediction was trained using T5-large (https://huggingface.co/docs/transformers/model_doc/t5)
on the TREC dataset (https://cogcomp.seas.upenn.edu/page/resource_view/49) .

See code/predict_T5.py for details on how to use our trained classifier based on T5-large.

We used Stanza (version 1.2) for obtaining dependency parses and NER information (https://stanfordnlp.github.io/stanza/)


The pkl dump of subsetted embeddings (using wordlist from Wiki) from GloVE can be created as follows:

--Use the make_embeddings function in  code/glove_similarity.py
with inputs the  original glove embeddings fromhttps://nlp.stanford.edu/data/glove.840B.300d.zip
and the terms list provided in code/wikiterms_stopwords.txt






