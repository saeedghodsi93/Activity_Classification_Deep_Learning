
    % reset
    clc
    clear all
    close all

    % functions path
    addpath(sprintf('%s',strcat(pwd,'\functions')));
    
    % dataset index 1:NTU 2:PKU
    dataset_idx = 1;
    
    load_dataset(dataset_idx);
    