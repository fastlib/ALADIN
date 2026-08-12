#arrhythmia specific benchmark
#python benchmark_delineation.py --method ALADIN --dataset VAL --perarrhythmia
#python benchmark_delineation.py --method Martinez --dataset VAL --perarrhythmia
#python benchmark_delineation.py --method Jiminez --dataset VAL --perarrhythmia

python benchmark_delineation.py --method ALADIN --dataset RDB --perarrhythmia
python benchmark_delineation.py --method Martinez --dataset RDB --perarrhythmia
python benchmark_delineation.py --method Jiminez --dataset RDB --perarrhythmia

#generate supplementary table 1
#python paper/generate_results_tables.py --dataset VAL
python paper/generate_results_tables.py --dataset RDB
