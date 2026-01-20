# FynnCode

Directory '_SSL_Model' contains checkpoint weights for LFNET model and training scripts and data preprocessing scripts

Directory '_SSL_Model_MFCC' contains the checkpoit weights for LFNET2D

# Instructions

'run_multiple_cmdarg_SingleChan_LFNET.sh' - execute this bash script to run LFNET. You will need to change the arguments to load data from your own directory.

*Note* Model weights have been pushed via git lfs

'run_model_trainer_SC_ONLY.py' - main python script

'run_model_trainer_HF_LF_SC.py' - main python script for hybrid model

'Ind_Table*' - indicies from noisy segment identification algorithm. Change them accordingly to suit your data. Run files in "Noisy_Segment_identification" directory to generate these .csv tables. MATLAB and python versions are both available.

## Add your files

* [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/MattBerry99/fynn_code.git
git branch -M main
git push -uf origin main
```
