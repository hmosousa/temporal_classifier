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
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135-a
python ../scripts/eval/semeval_eval.py -m hugosousa/smol-135-191329ff

# Create a conda environment with python 2.7
conda create -p ./.conda python=2.7 -y
conda activate ./.conda

# Run the evaluation
echo "Random"
python TE3-evaluation.py gold/ results/most_likely/random/
echo "Majority"
python TE3-evaluation.py gold/ results/most_likely/majority/
echo "Smol 135"
python TE3-evaluation.py gold/ results/most_likely/smol-135/
echo "Smol 135-a"
python TE3-evaluation.py gold/ results/most_likely/smol-135-a/
echo "Smol 135-191329ff"  # Best augmented model
python TE3-evaluation.py gold/ results/most_likely/smol-135-191329ff/
echo "hugosousa/smol-360-89128df1"  # Best augmented model
python TE3-evaluation.py gold/ results/most_likely/smol-360-89128df1/