# TiRank
A comprehensive RNA-seq and scRNA-seq integration analysis tool

### Install TiRank

We have provided three methods to install the TiRank package. 

Firstly, we highly recommend user to create a **new** conda environment and install `TiRank`

#### Locally pip install (Recommend)
```{bash}
  conda create -n -y TiRank

  conda activate TiRank

  git clone git@github.com:LenisLin/TiRank.git

  cd TiRank/TiRank_pack/TiRank

  pip install -e .
```

#### Locally conda install (Not recommend) (need more test)
```{bash}
git clone git@github.com:LenisLin/TiRank.git

## We will install TiRank via the environment file `TiRank.yml`
## You should firstly modified this file to replace the "perfix" in the bottom of this file with your path to the conda environment files

conda env create -f TiRank.yml
```

#### Docker (Recommend)

#### Web Server (Recommend)
