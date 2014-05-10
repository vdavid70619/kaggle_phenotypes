%%%
%%% Xiyang Dai
%%% dataloader for final project
%%% AS_IS: there is some performance issue when reading file line by line
%%%

function data = dataloader(varargin)

    CACHEFILE = 'data_cache.mat';
    
    for i = 1:length(varargin)
        switch lower(varargin{i})
            case {'datafile', '-df'}
                DATAFILE = varargin{i+1}; 
            case {'labelfile', '-lf'}
                LABLEFILE = varargin{i+1}; 
            case 'applyfun'
                APPLYFUN = varargin{i+1};
            case {'cachefile', '-cf'}
                CACHEFILE = varargin{i+1};                
        end
    end

    %% Load Data Line By Line
    id = {};
    feature = {};
    feature_label = {};
    population = {};

    if exist('DATAFILE')    
        fd = fopen(DATAFILE);
        fline = fgetl(fd);
        line = strsplit(fline,',');
        feature_label = line(2:end);
        i=1;
        fline = fgetl(fd);
        while ischar(fline)
            fprintf('Processing Line %d\n',i)
            line = strsplit(fline,',');
            fdim = length(line) - 1;
            line2feature = zeros(fdim,1);
            
            if exist('APPLYFUN')
                for j=1:fdim   
                    line2feature(j) = APPLYFUN(line{j+1})
                end
            else 
                for j=1:fdim       
                    switch line{j+1}
                        case '"A/A"'
                            line2feature(j) = 1;
                        case '"A/B"'
                            line2feature(j) = 2;
                        case '"B/A"'
                            line2feature(j) = 3;
                        case '"B/B"'
                            line2feature(j) = 4;
                        otherwise
                            line2feature(j) = 0;
                    end        
                end
            end
            id{i} = line{1};
            if ~isempty(strfind(line{1}, 'jpt'))
                population{i} = 1;
            elseif ~isempty(strfind(line{1}, 'ceu'))
                population{i} = 0;
            else
                population{i} = -1;
            end
                
            feature{i} = line2feature;
            fline = fgetl(fd);
            i = i+1;
        end
        fclose(fd);
    end 
    
    %% Load Labels
    label = {};
    
    if exist('LABLEFILE')
        fd = fopen(LABLEFILE);
        fline = fgetl(fd);
        line = strsplit(fline,',');
        header_line = line(2:end);
        id = {};
        i=1;
        fline = fgetl(fd);
        while ischar(fline)
            line = strsplit(fline,',');
            id{i} = line{1};
            label{i} = str2double(line{2});
            fline = fgetl(fd);
            i = i+1;
        end
        fclose(fd);    
    end  
    
    data.feature_label = feature_label;
    data.label = cat(1, label{:});
    data.feature = feature;
    data.id = id;
    data.population = cat(1, population{:});
    
    save(CACHEFILE, 'data')
end