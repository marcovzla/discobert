# discobert

    conda create -y -n discobert python=3.7
    conda activate discobert
    conda install -y ipython tqdm requests boto3 regex click joblib nltk scikit-learn jupyter
    conda install -y pytorch torchvision cpuonly -c pytorch
    pip install transformers
    
# discobert-gpu
    conda create -y -n discobert-gpu python=3.7
    conda activate discobert-gpu
    conda install -y ipython tqdm requests boto3 regex click joblib nltk scikit-learn jupyter
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    pip install transformers
    
