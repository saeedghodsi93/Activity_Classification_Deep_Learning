function [  ] = NTU_read_data( path )

    number_of_joints=25;
    number_of_dimensions=3;

    % read the missing file names
    missing_samples = textread('dataset\NTU_missing_samples.txt','%s','delimiter','\n');

    % init data map object
    data = containers.Map;

    files = dir(fullfile(path,'*.skeleton'));
    for file = 1:length(files)
        baseFileName = files(file).name;
        fullFileName = fullfile(path, baseFileName);

        % ignore the missing samples
        if any(find(strcmp(missing_samples,baseFileName(1:20))))
            continue;
        end

        % open the file and read
        fileid = fopen(fullFileName);
        framecount = fscanf(fileid,'%d',1); % no of the recorded frames
        bodyinfo=[]; % to store multiple skeletons per frame
        for f=1:framecount
            bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
            for b=1:bodycount
                clear body;
                body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
                arrayint = fscanf(fileid,'%d',6); % read 6 integers
                body.clipedEdges = arrayint(1);
                body.handLeftConfidence = arrayint(2);
                body.handLeftState = arrayint(3);
                body.handRightConfidence = arrayint(4);
                body.handRightState = arrayint(5);
                body.isResticted = arrayint(6);
                lean = fscanf(fileid,'%f',2);
                body.leanX = lean(1);
                body.leanY = lean(2);
                body.trackingState = fscanf(fileid,'%d',1);

                body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
                for j=1:body.jointCount
                    jointinfo = fscanf(fileid,'%f',11);
                    joint=[];

                    % 3D location of the joint j
                    joint.x = jointinfo(1);
                    joint.y = jointinfo(2);
                    joint.z = jointinfo(3);

                    % 2D location of the joint j in corresponding depth/IR frame
                    joint.depthX = jointinfo(4);
                    joint.depthY = jointinfo(5);

                    % 2D location of the joint j in corresponding RGB frame
                    joint.colorX = jointinfo(6);
                    joint.colorY = jointinfo(7);

                    % The orientation of the joint j
                    joint.orientationW = jointinfo(8);
                    joint.orientationX = jointinfo(9);
                    joint.orientationY = jointinfo(10);
                    joint.orientationZ = jointinfo(11);

                    % The tracking state of the joint j
                    joint.trackingState = fscanf(fileid,'%d',1);

                    body.joints(j)=joint;
                end
                bodyinfo(f).bodies(b)=body;
            end
        end
        fclose(fileid);

        % determine all the tracked subjects
        ids = {};
        for f = 1:length(bodyinfo)
            for s = 1:length(bodyinfo(f).bodies)
                id = int2str(bodyinfo(f).bodies(s).bodyID);
                if ~any(ismember(id,ids))
                    ids{length(ids)+1} = id;
                end
            end
        end

        % store the bodies in a map object
        bodies = containers.Map;
        for s = 1:length(ids)
            body = NaN(length(bodyinfo),number_of_joints,number_of_dimensions);
            bodies(ids{s}) = body;
        end
        for f = 1:length(bodyinfo)
            for s = 1:length(bodyinfo(f).bodies)
                id = int2str(bodyinfo(f).bodies(s).bodyID);
                skeleton = bodies(id);
                for j = 1:number_of_joints
                    skeleton(f,j,1) = bodyinfo(f).bodies(s).joints(j).x;
                    skeleton(f,j,2) = bodyinfo(f).bodies(s).joints(j).y;
                    skeleton(f,j,3) = bodyinfo(f).bodies(s).joints(j).z;
                end
                bodies(id) = skeleton;
            end
        end

        % store the bodies map to data map
        data(baseFileName) = bodies;

        % log
        ks = keys(data);
        if mod(length(ks),100)<1
            display(length(ks));
        end
                
    end

    save('dataset\NTU_Data.mat','data','-v7.3');

end
