function [Maps] = auto_stitch(regdir)
%% Combine PLM images using automatic feature matching

% Check if 'lfolder' exists, if not, set to current directory
if ~exist('lfolder', 'var')
    lfolder = pwd;
end

% Set the stitching parameter and load order
stch_param = 'PI'; % or 'PI' or 'AZ'
if nargin == 0
    regdir = 'right';
end

if strcmp(regdir, 'left')
    loadOrder = 'descend';
else
    loadOrder = 'ascend';
end

% Get files
[FileName, PathName] = uigetfile(fullfile(lfolder, '*.mat'), 'Select the .mat-files', 'MultiSelect', 'on');

if ~exist('sfolder', 'var')
    sfolder = '.';
    SavePath = fullfile(PathName, sfolder);
    if ~exist(SavePath, 'dir')
        mkdir(SavePath);
    end
end

if ~iscell(FileName)
    error('Select at least 2 files.');
end

% Sort files based on numeric suffix
fileName = string(FileName);
numericSuffixes = zeros(length(FileName), 1);
for i = 1:length(FileName)
    tokens = regexp(FileName{i}, '_(\d+)\.mat$', 'tokens');
    if ~isempty(tokens)
        numericSuffixes(i) = str2double(tokens{1}{1});
    else
        numericSuffixes(i) = NaN;
    end
end
[~, sortedIndices] = sort(numericSuffixes, loadOrder);
fileName = fileName(sortedIndices);

% Convert string array back to cell array
FileName = cellstr(fileName);

% Load images
switch stch_param
    case {'AZ', 'AZmax', 'PI'}
        % First pass: Determine global max dimensions across all files and variables
        max_rows = 0;
        max_cols = 0;

        for ii = 1:length(FileName)
            data = load(fullfile(PathName, FileName{ii}), 'Maps');
            data = data.Maps;
            current_AZ = squeeze(data.AZ);
            
            % Update max dimensions for this file
            current_max_rows = size(current_AZ, 1);
            current_max_cols = size(current_AZ, 2);
            
            % Update global max
            max_rows = max(max_rows, current_max_rows);
            max_cols = max(max_cols, current_max_cols);
        end

        % Initialize 3D arrays with global max dimensions
        AZ = zeros(max_rows, max_cols, length(FileName));
        AZmax = zeros(max_rows, max_cols, length(FileName));
        PI = zeros(max_rows, max_cols, length(FileName));
        
        % Second pass: Load and pad each variable to global max size
        for ii = 1:length(FileName)
            data = load(fullfile(PathName, FileName{ii}), 'Maps');
            data = data.Maps;
            current_AZ = squeeze(data.AZ);
            current_AZmax = squeeze(data.AZmax);
            current_PI = squeeze(data.PI(:,:,1));
            
            % Pad each variable to max_rows and max_cols with zeros (post-padding)
            pad_AZ = [max_rows - size(current_AZ,1), max_cols - size(current_AZ,2)];
            padded_AZ = padarray(current_AZ, pad_AZ, 0, 'post');
            
            pad_AZmax = [max_rows - size(current_AZmax,1), max_cols - size(current_AZmax,2)];
            padded_AZmax = padarray(current_AZmax, pad_AZmax, 0, 'post');
            
            pad_PI = [max_rows - size(current_PI,1), max_cols - size(current_PI,2)];
            padded_PI = padarray(current_PI, pad_PI, 0, 'post');
            
            % Assign padded data to 3D arrays
            AZ(:,:,ii) = double(padded_AZ);
            AZmax(:,:,ii) = double(padded_AZmax);
            PI(:,:,ii) = double(padded_PI);
        end

    case 'dd'
        AZ = [];
        AZmax = [];
        PI = [];
        for ii = 1:length(FileName)
            load(fullfile(PathName, char(FileName{ii})));
            AZ(:,:,ii) = Image_OD;
            AZmax(:,:,ii) = Image_OD;
            PI(:,:,ii) = Image_OD;
        end
end

% Name for saving
Name = string(FileName{1});
if exist(sprintf('%s%s', Name(1:end-8), 'manu.mat'), 'file')
    Name = sprintf('%s%s%s', Name(1:end-8), 'man', round(rand() * 10));
end

% Select the image to display based on stch_param
switch stch_param
    case {'PI', 'dd'}
        ShowIm = PI;
    case 'AZ'
        ShowIm = abs(AZ);
    case 'AZmax'
        ShowIm = abs(AZmax);
end

numImages = size(ShowIm, 3);
fprintf('%d images are selected. \n', numImages);

% Initialize cumulative transformations
tforms(numImages) = affine2d(eye(3));
imageSize = zeros(numImages,2);

% Initialize xlim and ylim limits
xlimTotal = [Inf, -Inf];
ylimTotal = [Inf, -Inf];

% Process first image
imageSize(1,:) = size(ShowIm(:,:,1));
[xlim, ylim] = outputLimits(tforms(1), [1 imageSize(1,2)], [1 imageSize(1,1)]);
xlimTotal = [min(xlimTotal(1), xlim(1)), max(xlimTotal(2), xlim(2))];
ylimTotal = [min(ylimTotal(1), ylim(1)), max(ylimTotal(2), ylim(2))];

% Automatic stitching using SURF feature matching
for ii = 2:numImages
    fprintf('Registering image %d to image %d \n', ii, ii-1)

    % Get fixed and moving images
    FIXED = ShowIm(:,:,ii-1);
    MOVING = ShowIm(:,:,ii);

    % Scale the images to make the feature detection more efficient
    scaleFactor = 1; % Adjust based on image size

    % Resize images to smaller scale
    FIXED_small = imresize(FIXED, scaleFactor);
    MOVING_small = imresize(MOVING, scaleFactor);

    % Normalize images
    FIXED_norm = adapthisteq(mat2gray(FIXED_small));
    MOVING_norm = adapthisteq(mat2gray(MOVING_small));

    % % Detect ORB features
    % fixedPoints = detectORBFeatures(FIXED_norm, 'ScaleFactor', 1.2, 'NumLevels', 8);
    % movingPoints = detectORBFeatures(MOVING_norm, 'ScaleFactor', 1.2, 'NumLevels', 8);
    % % Extract features
    % [fixedFeatures, fixedValidPoints] = extractFeatures(FIXED_norm, fixedPoints);
    % [movingFeatures, movingValidPoints] = extractFeatures(MOVING_norm, movingPoints);

    % % Detect KAZE features
    % fixedPoints = detectKAZEFeatures(FIXED_norm, 'Diffusion', 'region', 'Threshold', 0.0001,'NumOctaves',4,'NumScaleLevels',7);
    % movingPoints = detectKAZEFeatures(MOVING_norm, 'Diffusion', 'region', 'Threshold', 0.0001,'NumOctaves',4,'NumScaleLevels',7);
    % % Extract features
    % [fixedFeatures, fixedValidPoints] = extractFeatures(FIXED_norm, fixedPoints, 'Upright', false);
    % [movingFeatures, movingValidPoints] = extractFeatures(MOVING_norm, movingPoints, 'Upright', false);

    % Detect SURF features
    fixedPoints = detectSURFFeatures(FIXED_norm, 'MetricThreshold', 1000, 'NumOctaves', 4, 'NumScaleLevels', 6);
    movingPoints = detectSURFFeatures(MOVING_norm, 'MetricThreshold', 1000, 'NumOctaves', 4, 'NumScaleLevels', 6);
    % Extract features
    [fixedFeatures,fixedValidPoints] = extractFeatures(FIXED_norm, fixedPoints, 'Upright', false);
    [movingFeatures,movingValidPoints] = extractFeatures(MOVING_norm, movingPoints, 'Upright', false);

    % Match features
    indexPairs = matchFeatures(fixedFeatures, movingFeatures, 'MatchThreshold', 20, 'MaxRatio', 0.20);
    fixedMatchedPointsSmall = fixedValidPoints(indexPairs(:,1));
    movingMatchedPointsSmall = movingValidPoints(indexPairs(:,2));

    % Scale points back to original image size
    fixedMatchedPoints = fixedMatchedPointsSmall.Location / scaleFactor;
    movingMatchedPoints = movingMatchedPointsSmall.Location / scaleFactor;

    % Check if enough matched points
    if size(fixedMatchedPoints, 1) < 3
        fprintf(['Not enough matched points to register image %d to image %d. \n' ...
            'Select the matching points manually. \n'], ii, ii-1);

        % Manual selection of points
        width = 30;
        iterations = 2;
        Point_1 = zeros(iterations, 2);
        Point_2 = zeros(iterations, 2);
        for iteration = 1:iterations
            colormap jet
            axis_1 = [1,size(ShowIm,1),1,size(ShowIm,2)];
            subplot(1,2,1); imagesc(FIXED);
            title("Select point " + num2str(iteration) + " in Image 1");
            axis_2 = [1,size(ShowIm,1),1,size(ShowIm,2)];
            subplot(1,2,2); imagesc(MOVING);
            title("Select point "+num2str(iteration)+" in Image 2");
            [x,y] = ginput(2);
            % New axes for fig 1
            axis_1U = round([axis_1(1)+y(1)-width/2,axis_1(1)+y(1)+width/2,axis_1(3)+x(1)-width/2,axis_1(3)+x(1)+width/2]);
            % Checking axes
            if axis_1U(4) > size(ShowIm,2)
                axis_1U(4) = size(ShowIm,2);
                axis_1U(3) = size(ShowIm,2)-width;
            end
            subplot(1,2,1); imagesc(FIXED(axis_1U(1):axis_1U(2),axis_1U(3):axis_1U(4)));
            title("Refine point "+num2str(iteration)+" in Zoomed Image 1");
            % New axes for fig 2
            axis_2U = round([axis_2(1)+y(2)-width/2,axis_2(1)+y(2)+width/2,axis_2(3)+x(2)-width/2,axis_2(3)+x(2)+width/2]);
            % Checking axes
            if axis_2U(3) < 1
                axis_2U(3) = 1;
                asix_2U(4) = axis_2U(3)+width;
            end
            subplot(1,2,2); imagesc(MOVING(axis_2U(1):axis_2U(2),axis_2U(3):axis_2U(4)));
            title("Refine point "+num2str(iteration)+" in Zoomed Image 2");
            clear x y
            [x,y] = ginput(2);
            Point_1(iteration, :) = round([axis_1U(3)+x(1),axis_1U(1)+y(1)]);
            Point_2(iteration, :) = round([axis_2U(3)+x(2),axis_2U(1)+y(2)]);
        end
        close(gcf)

        fixedMatchedPoints = Point_1;
        movingMatchedPoints = Point_2;
    end

    % Estimate geometric transformation
    tformEstimate = estimateGeometricTransform2D(movingMatchedPoints, fixedMatchedPoints, 'similarity');

    % figure;
    % showMatchedFeatures(FIXED, MOVING, movingMatchedPoints, fixedMatchedPoints, 'montage');

    % Compute cumulative transformation
    tforms(ii).T = tformEstimate.T * tforms(ii-1).T;

    % Get image size
    imageSize(ii,:) = size(ShowIm(:,:,ii));

    % Compute output limits
    [xlim, ylim] = outputLimits(tforms(ii), [1 imageSize(ii,2)], [1 imageSize(ii,1)]);

    % Update total xlim and ylim
    xlimTotal = [min(xlimTotal(1), xlim(1)), max(xlimTotal(2), xlim(2))];
    ylimTotal = [min(ylimTotal(1), ylim(1)), max(ylimTotal(2), ylim(2))];
end

% Compute the size of the panorama.
xMin = min([1, xlimTotal(1)]);
xMax = max([imageSize(1,2), xlimTotal(2)]);
yMin = min([1, ylimTotal(1)]);
yMax = max([imageSize(1,1), ylimTotal(2)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

% Create an output reference object
xWorldLimits = [xMin xMax];
yWorldLimits = [yMin yMax];
panoramaView = imref2d([height width], xWorldLimits, yWorldLimits);


% Initialize cumulative weighted sum and cumulative weight
cumulativeWeightedSumAZ = zeros([height, width]);
cumulativeWeightAZ = zeros([height, width]);

cumulativeWeightedSumAZmax = zeros([height, width]);
cumulativeWeightAZmax = zeros([height, width]);

cumulativeWeightedSumPI = zeros([height, width]);
cumulativeWeightPI = zeros([height, width]);

% Create the panorama with weighted averaging
for ii = 1:numImages
    % Transform images into panoramaView coordinate frame
    transformedImageAZ = imwarp(AZ(:,:,ii), tforms(ii), 'OutputView', panoramaView);
    transformedImageAZmax = imwarp(AZmax(:,:,ii), tforms(ii), 'OutputView', panoramaView);
    transformedImagePI = imwarp(PI(:,:,ii), tforms(ii), 'OutputView', panoramaView);
    
    % Create weight mask that decreases slowly over the y-axis
    [rows, cols] = size(AZ(:,:,ii));
    [X, Y] = meshgrid(1:cols, 1:rows);

    centerX = cols ./ 2;
    centerY = rows ./ 2;

    offsetX = 0.10; % ratio of cols to offset centerX towards left positions
    offsetY = 0.00; % ratio of rows to offset centerY towards upper positions
    
    centerX = centerX - offsetX .* cols;
    centerY = centerY - offsetY .* rows;

    distanceFromCenterX = X - centerX;
    distanceFromCenterY = Y - centerY;
    
    % Set sigmaX and sigmaY to control the rate of decrease over x and y
    sigmaX = cols / 10; % Adjust sigmaX as needed
    sigmaY = rows / 2;
    
    % Compute the weight mask
    weightMask = exp(- ( (distanceFromCenterX.^2) / (2 * sigmaX^2) + (distanceFromCenterY.^2) / (2 * sigmaY^2) ));
    
    % Transform weight mask into panoramaView coordinate frame
    transformedWeightMask = imwarp(weightMask, tforms(ii), 'OutputView', panoramaView);
    
    % Create mask for non-zero pixels in transformed image
    mask = transformedImageAZ ~= 0;
    
    % Set transformed weight mask to zero where transformed image is zero
    transformedWeightMask(~mask) = 0;
    
    % Update cumulative weighted sum and cumulative weight for AZ
    cumulativeWeightedSumAZ = cumulativeWeightedSumAZ + transformedImageAZ .* transformedWeightMask;
    cumulativeWeightAZ = cumulativeWeightAZ + transformedWeightMask;
    
    % Similarly for AZmax
    cumulativeWeightedSumAZmax = cumulativeWeightedSumAZmax + transformedImageAZmax .* transformedWeightMask;
    cumulativeWeightAZmax = cumulativeWeightAZmax + transformedWeightMask;
    
    % Similarly for PI
    cumulativeWeightedSumPI = cumulativeWeightedSumPI + transformedImagePI .* transformedWeightMask;
    cumulativeWeightPI = cumulativeWeightPI + transformedWeightMask;
end

% Compute final combined images
CombinedImageAZ = cumulativeWeightedSumAZ ./ cumulativeWeightAZ;
CombinedImageAZ(isnan(CombinedImageAZ)) = 0;

CombinedImageAZmax = cumulativeWeightedSumAZmax ./ cumulativeWeightAZmax;
CombinedImageAZmax(isnan(CombinedImageAZmax)) = 0;

CombinedImagePI = cumulativeWeightedSumPI ./ cumulativeWeightPI;
CombinedImagePI(isnan(CombinedImagePI)) = 0;

% Optimize window by cropping zero rows and columns
Index = find(mean(CombinedImagePI, 2) > 0);
Index2 = find(mean(CombinedImagePI) > 0);
CombinedImagePI = CombinedImagePI(min(Index):max(Index), min(Index2):max(Index2));
CombinedImageAZ = CombinedImageAZ(min(Index):max(Index), min(Index2):max(Index2));
CombinedImageAZmax = CombinedImageAZmax(min(Index):max(Index), min(Index2):max(Index2));

% Display the combined image
if size(CombinedImagePI, 1) < size(CombinedImagePI, 2)
    Limit_2 = size(CombinedImagePI, 1) / size(CombinedImagePI, 2) * (1920 / 1080);
    Limit_1 = size(CombinedImagePI, 2) / size(CombinedImagePI, 2);
else
    Limit_1 = size(CombinedImagePI, 2) / size(CombinedImagePI, 1);
    Limit_2 = size(CombinedImagePI, 1) / size(CombinedImagePI, 1);
end

h = figure('units', 'normalized', 'outerposition', [0 0 Limit_1 Limit_2]);
switch stch_param
    case 'AZ'
        imagesc(CombinedImageAZ);
    case 'AZmax'
        imagesc(CombinedImageAZmax)
    case 'PI'
        imagesc(CombinedImagePI);
end

set(gca, 'position', [0 0 1 1], 'units', 'normalized');
set(h, 'PaperPositionMode', 'auto');
axis off;
colormap jet;
colorbar;

% Prompt to save the stitched data
Name = char(Name);
answer = questdlg('Save stitched data?', 'Save', 'Yes', 'Try Again', 'Cancel', 'Cancel');

% Handle response
save_switch = 1;
switch answer
    case 'Yes'
        clear data
        Maps.AZ = CombinedImageAZ;
        Maps.AZmax = CombinedImageAZmax;
        Maps.PI = CombinedImagePI;
        name = char(Name);
        PathName_save = fullfile(PathName, sfolder, [Name(1:end-6), '_combined.mat']);
        save(PathName_save, 'Maps');
    case 'Try Again'
        disp('Data not saved, trying again.');
        % You can implement a retry mechanism here if needed
    case 'Cancel'
        save_switch = 0;
        disp('Operation cancelled, data not saved.');
end

if save_switch
    disp('Data saved successfully.');
end

end
