#ResNet embeddings
python embedder.py --method reference --datatype train
python embedder.py --method reference --datatype id_test
python embedder.py --method reference --datatype ood_test

#ALADIN embeddings
python embedder.py --method aladin --datatype train
python embedder.py --method aladin --datatype id_test
python embedder.py --method aladin --datatype ood_test