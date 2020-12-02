function [  ] = NTU_ask_uncertain(  )

    % read the data
    load('dataset\NTU_Data.mat','data');
    
    % read the uncertain file names
    uncertain = textread('dataset\NTU_uncertain_samples.txt','%s','delimiter','\n');

    % read the actions names
    action_names = textread('dataset\NTU_action_names.txt','%s','delimiter','\n');

    % read the valid body numbers map object
    load('dataset\NTU_Uncertain.mat','valid_bodies');

    for file = 1:length(uncertain)
        baseFileName = uncertain(file);
        baseFileName = baseFileName{1};

        % skip the previousely asked samples
        if ~all(valid_bodies(baseFileName)==0)
            continue;
        end

        % file parameters
        action = str2num(baseFileName(18:20));

        % extract the bodies data
        bodies = data(baseFileName);
        
        % show the bodies
        display(action_names{action});
        draw_skeleton(bodies,10)

        % read the user input
        prompt = 'enter valid bodies: ';
        comm = input(prompt);

        % check the input
        while comm==0
            draw_skeleton(bodies,10)
            prompt = 'enter valid bodies: ';
            comm = input(prompt);
        end

        if comm < 0
            break;
        else
            % store the indices in the map object
            valid_bodies(baseFileName) = comm;
        end

    end

    save('dataset\NTU_Uncertain.mat','valid_bodies');
    
    % init the valid body numbers map object
%     valid_bodies = containers.Map;
%     for u = 1:length(uncertain)
%         valid_bodies(uncertain{u}) = 0;
%     end
%     cd('dataset')
%     save('NTU_Uncertain.mat','valid_bodies');
%     cd('..')

end
