import os
TEST = ['Bacillus','Listeria','Salmonella','ecoli','sue','hide','ccu063']

for test in TEST:
    os.chdir('{test}'.format(test=test))
    os.system('python ../deletion_predict.py insertion_homo_alignment.h5 ../SVM_insertion_homo_prob.model>threshold.txt')
    #os.system('python ../prob_npz.py insertion_homo_alignment.h5 ../SVM_insertion_homo_prob.model')
    os.chdir('../')
