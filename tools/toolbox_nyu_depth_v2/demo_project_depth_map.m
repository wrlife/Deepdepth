% Demos the use of project_depth_map.m

% % The location of the RAW dataset.
% CLIPS_DIR = '[]';
% 
% % The path to the labeled dataset.
% LABELED_DATASET_PATH = '[]';
% 
% load(LABELED_DATASET_PATH, 'rawDepthFilenames', 'rawRgbFilenames');

datapath='../../data/nyud/basements/basement_0001a/';
outpath='../../data/nyud/processed/';

fid=fopen(fullfile(datapath,'INDEX.txt'));

tline=fgetl(fid);
count=1;
while ischar(tline)
    
    depthname=tline;
    fgetl(fid);
    rgbname=fgetl(fid);

    %% Load a pair of frames and align them.
    imgRgb = imread(sprintf('%s%s', datapath,rgbname));

    imgDepth = imread(sprintf('%s%s', datapath,depthname));
    imgDepth = swapbytes(imgDepth);

    [imgDepth2, imgRgb2] = project_depth_map(imgDepth, imgRgb);
    
    depthname=sprintf('basement0001a_d%06d.bin',count);
    imgname=sprintf('basement0001a_r%06d.jpg',count);
    
    fiddepth=fopen(fullfile(outpath,depthname),'wb');
    fwrite(fiddepth,imgDepth2,'float32');
    fclose(fiddepth);
    
    imwrite(imgRgb2,fullfile(outpath,imgname));
    
    count=count+1;

    %% Now visualize the pair before and after alignment.
%     imgDepthAbsBefore = depth_rel2depth_abs(double(imgDepth));
%     imgOverlayBefore = get_rgb_depth_overlay(imgRgb, imgDepthAbsBefore);
% 
%     imgOverlayAfter = get_rgb_depth_overlay(imgRgb2, imgDepth2);
% 
%     figure;
%     subplot(1,2,1);
%     imagesc(crop_image(imgOverlayBefore));
%     title('Before projection');
% 
%     subplot(1,2,2);
%     imagesc(crop_image(imgOverlayAfter));
%     title('After projection');
    
end
fclose(fid);

