
workPath = pwd;
dataPath = fullfile(workPath, 'data');
contentList = dir(dataPath);

imgOutPath = "data_img";
if ~exist(imgOutPath, 'dir')==1
    mkdir(imgOutPath)
end

classList = [];
for idx = 3:length(contentList)
    if contentList(idx).isdir
        classList = [classList, string(contentList(idx).name)];
    end
end

for class_idx = 1:length(classList)
    class = classList(class_idx);
    disp(strcat('Converting Object: ',class));
    classPath = fullfile(dataPath,class);
    classOutPath = fullfile(imgOutPath,class);
    if ~exist(classOutPath, 'dir')==1
        mkdir(classOutPath)
    end
    frameList = dir(classPath);
    num_frame = length(frameList)-2;

    msg = 0;
    for fr_idx = 3:length(frameList)
        % Show progress in console
        progress = 100*(fr_idx-2)/num_frame;
        fprintf(1,repmat('\b',1,msg));
        msg = fprintf(1,'-- Percent done: %.2f%%', progress);
        
        if endsWith(frameList(fr_idx).name, '.aedat')
            frameName = frameList(fr_idx).name(1:end-6);
            dataFile = char(fullfile(dataPath,class,frameList(fr_idx).name));
            data = dat2mat(dataFile);

            frame = zeros(128,128);
            data(:,4:5) = data(:,4:5)+1;

            max_pxl = data(end,1);
            min_pxl = data(1,1);

            data(:,1) = data(:,1)./(max_pxl-min_pxl);
            for i = 1:size(data,1)
                x = data(i,4);
                y = data(i,5);
                frame(x,y) = data(i,1);
            end
            
            % Save as .png
            imgName = strcat(frameName,'.png');
            imgPath = fullfile(classOutPath, imgName);
            % imshow(frame);
            imwrite(frame, imgPath);
        end
    end
    fprintf('\n');
end