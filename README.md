# Quinn: Complex Word Identification using Neural Networks (CWI-NN)

Information available on the [Shared Task (2018) website](http://sites.google.com/view/cwisharedtask2018/).

### About Model

Bi-directional RNN (GRU) with "masked" soft-attention written in TensorFlow. The attention mask is decided from the annotator specified context (see dataset).

### Steps

- Download the Shared Task dataset from the website.

- Download GloVe dataset from [here](https://nlp.stanford.edu/projects/glove/) and copy into respective directories.

- Generate embeddings and vocabulary with: `python utils/generate_embeddings.py -d ./data/embeddings/glove.6B.300d.txt --npy_output ./data/dumps/embeddings.npy --dict_output ./data/dumps/vocab.pckl --dict_whitelist ./data/embeddings/vocab.txt`

- Train with `python train.py`

**TO BE DONE:** Argument parser for train.py; Write test.py with F1 scores and other relevant evaluation metrics.

### References

J. Pennington, R. Socher and C. D. Manning, [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), 2014.

N. S. Hartmann and L. B. dos Santos, [NILC at CWI 2018: Exploring Feature Engineering and Feature Learning](http://aclweb.org/anthology/W18-0540), 2018.

N. Gillin, [Sensible at SemEval-2016 Task 11: Neural Nonsense Mangled in Ensemble Mess](http://www.aclweb.org/anthology/S16-1148), 2016.

### Sources

Embeddings helper: [rampage644/qrnn](https://github.com/rampage644/qrnn)

Miscellaneous: [dennybritz/cnn-text-classification-tf/](https://github.com/dennybritz/cnn-text-classification-tf/)

---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
