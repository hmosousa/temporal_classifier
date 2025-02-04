# Clone the Tempeval toolkit repository
git clone https://github.com/naushadzaman/tempeval3_toolkit.git
cd tempeval3_toolkit


# Download the dataset
wget https://figshare.com/ndownloader/files/17225243 -O te3.tar.bz2
tar -xvf te3.tar.bz2
rm te3.tar.bz2
tar -xvf TempEval-3/DATA-PUBLISHED/TimeML-Platinum-ADCR2013T001.tar.gz

# Run the annotation
python ../scripts/eval/semeval_eval.py -m random
python ../scripts/eval/semeval_eval.py -m majority
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135-0dd0da37 -r aadb93fe6ee0272e40eac187511de316afa94b5b
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135-a-191329ff -r bc857f718f69adb90ea7baf674d297c35867f6c6
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135-c-3ed00d05 -r 73a6cca7b0b814e774987594820de75a6aab2f33
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135-ac-a4eaad65 -r 34302bb3e114e8650b3ab60ad35d0befad00c04c

python ../scripts/eval/semeval_eval.py -m hugosousa/smol-360-89128df1 -r edd5ea745fa2d4aebfaf9a4e576ec594dcd840ba
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-360-a-4a820490 -r aee726c847b87a0fe8123f66a6f374fee5bbece5

# Create a conda environment with python 2.7
conda create -p ./.conda python=2.7 -y
conda activate ./.conda

# Run the evaluation
echo "Random" && python TE3-evaluation.py gold/ results/most_likely/random/
echo "Majority" && python TE3-evaluation.py gold/ results/most_likely/majority/

echo "Smol 135-r"  && python TE3-evaluation.py gold/ results/most_likely/smol-135-0dd0da37
echo "Smol 135-a"  && python TE3-evaluation.py gold/ results/most_likely/smol-135-a-191329ff
echo "Smol 135-c"  && python TE3-evaluation.py gold/ results/most_likely/smol-135-c-3ed00d05
echo "Smol 135-ac" && python TE3-evaluation.py gold/ results/most_likely/smol-135-ac-a4eaad65

echo "Smol 360-89128df1"   && python TE3-evaluation.py gold/ results/most_likely/smol-360-89128df1
echo "Smol-360-a-4a820490" && python TE3-evaluation.py gold/ results/most_likely/smol-360-a-4a820490