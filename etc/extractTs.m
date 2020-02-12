


% Load in the data matricies
%/Users/K/Desktop/fortinLab/data_playground/Data/
%load('/Users/K/Desktop/fortinLab/Data/Mitt_July18_5odorswithSkips_EnsembleMatrix.mat');
%load('/Users/K/Desktop/fortinLab/Data/Mitt_July18_5odorswithSkips_BehaviorMatrix.mat');

% let's navigate to the data directory, wherever that may be...
% Path = '/Volumes/PNY64_101/Mitt';
%Path = fullfile('D:','Documents','Data','superchris','02122009 FamiliarSeq Skips');
%Path = fullfile('D:','Documents','Data','superchris','superchris_drive');

% takes about 15.5833 minutes to run... 
Path = fullfile('D:','Documents','Data','driveCA1_wellTrained');
rat_name = {'Barat', 'Buchanan', 'Stella', 'Superchris', 'Mitt'};

t_all = tic;
for ii = 1:length(rat_name)
    ratPath = fullfile(Path, rat_name{ii}, filesep);

    cd(ratPath);

    %%
    % As per what Lingge had used to transform the data, the EpochExtraction_SM function
    % (for example see: Phase Position Scatter CA1 PROTO.m)
    disp(rat_name{ii})
    fprintf("\nExtracting files...\n")
    events = {'PokeIn', 'PokeOut'};
    
    for e_i = 1:length(events)
        disp(events{e_i})
        t_rat = tic;
        
        [unitEpoch,...
         unitIDs,...
         lfpEpoch,... 
         lfpIDs,...
         trialTimeBins,...
         eventTimeBins,...
         trialInfo] = EpochExtraction_SM(events{e_i}, -2, 2);
        % , 'lfpBand', 'All'
        
        toc(t_rat)

        % Great! Now let's save our hard work
        fprintf("Saving file: \n")
        %saveName = "mitt_extraction_odor2s.mat";
        savePath = fullfile('D:','Documents','Data', 'extracted_CA1_trials', filesep);
        saveName = [savePath, rat_name{ii}, '_extraction_odor2s_', events{e_i}, '.mat'];
        save(saveName,"unitEpoch", "unitIDs", "lfpEpoch", "lfpIDs", "trialTimeBins", "eventTimeBins", "trialInfo")
        % , '-v7.3' '-v6' % Removed the , '-v7.3' flag from the save file last arg
        disp(saveName);
        
    end
end
toc(t_all)
disp('fin')




%% BONEYARD
% {'Barat', 'Buchanan', 'Stella', 'Superchris', 'Mitt'}
pitimes = [68.053727, 86.760797, 114.112472, 63.564484, 181.467804];
potimes = [68.151123, 61.607692, 55.482607, 57.492072, 114.554434];
figure; plot(1:5, pitimes, 1:5, potimes)


if 0 % Not used
    % Now let's organize these guy's behavior by their trials:
    preTrialBehavMatrix = OrganizeTrialData_SM(behavMatrix, behavMatrixColIDs, [0 1.5], 'PokeIn');

    % And now let's fix up the cells...
    preTrialEnsemble = ExtractTrialData_SM(preTrialBehavMatrix, ensembleMatrix(:,2:end)); %#ok<*NODEF>`
end
