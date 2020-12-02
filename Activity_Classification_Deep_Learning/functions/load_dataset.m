function [  ] = load_dataset( dataset_idx )

    % load or reload dataset, 0:load dataset, 1:reload dataset(time consuming)
    reload_idx=4;
    
    % load skeleton matrix from database
    switch dataset_idx
       
        case 1
            path='dataset\Skeleton';
            base_joints=[1,5,9];
            load_NTU_dataset(path,base_joints,reload_idx);
            
    end
    
end
