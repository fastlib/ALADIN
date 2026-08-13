#dataset CINC
python benchmark_diagnosis.py --method ALADIN --dataset CINC --overwrite
python benchmark_diagnosis.py --method Hannun --dataset CINC --overwrite --modelpaths "/data/benchmark/weights/HannunNet_checkpoint_best.pth"
python benchmark_diagnosis.py --method ECGFounder --dataset CINC --overwrite --modelpaths "/data/benchmark/weights/ECGFounderNet_checkpoint_best.pth"
