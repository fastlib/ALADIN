#dataset STANFORD
python benchmark_diagnosis.py --method ALADIN --dataset STANFORD --overwrite 
python benchmark_diagnosis.py --method Hannun --dataset STANFORD --overwrite --modelpaths ["/data/benchmark/weights/HannunNet_checkpoint_best.pth"]
python benchmark_diagnosis.py --method ECGFounder --dataset STANFORD --overwrite --modelpaths ["/data/benchmark/weights/1_lead_ECGFounder.pth"]

