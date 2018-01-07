clear;
close all;
folder = 'datasets';

savepath = 'train.h5';

size_label = 192;
scale = 8;
size_input = size_label/scale;
size_x2 = size_label/4;
size_x4 = size_label/2;
stride = 128;
%% downsizing
downsizes = [1,0.7,0.5];

data = zeros(size_input, size_input, 3, 1);
label_x2 = zeros(size_x2, size_x2, 3, 1);
label_x4 = zeros(size_x4, size_x4, 3, 1);
label_x8 = zeros(size_label, size_label, 3, 1);

bicubic_x2 = zeros(size_x2, size_x2, 3, 1);
bicubic_x4 = zeros(size_x4, size_x4, 3, 1);
bicubic_x8 = zeros(size_label, size_label, 3, 1);
count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];


length(filepaths)

tic
for i = 1 : length(filepaths)
    for downsize = 1 : length(downsizes)
        image = imread(fullfile(folder,filepaths(i).name));
        image = imresize(image,downsizes(downsize),'bicubic');
        
        for flip = 1: 3
            if flip == 1
                image = flipdim(image ,1);
            end
            if flip == 2
                image = flipdim(image ,2);
            end
            
            for degree = 1 : 4
                image = imrotate(image, 90 * (degree - 1));
                if size(image,3)==3
                    image = im2double(image);
                    im_label = modcrop(image, scale);
                    [hei,wid, c] = size(im_label);

                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            subim_label_x2 = imresize(subim_label,1/2,'bicubic');
                            subim_label_x4 = imresize(subim_label,1/4,'bicubic');
                            subim_input = imresize(subim_label,1/scale,'bicubic');

                            subim_bicubic_x2 = imresize(subim_input,2,'bicubic');
                            subim_bicubic_x4 = imresize(subim_input,4,'bicubic');
                            subim_bicubic_x8 = imresize(subim_input,8,'bicubic');

                            count=count+1;
                            data(:, :, :, count) = subim_input;            
                            label_x2(:, :, :, count) = subim_label_x4;  
                            label_x4(:, :, :, count) = subim_label_x2;   
                            label_x8(:, :, :, count) = subim_label;

                            bicubic_x2(:, :, :, count) = subim_bicubic_x2;
                            bicubic_x4(:, :, :, count) = subim_bicubic_x4;
                            bicubic_x8(:, :, :, count) = subim_bicubic_x8;
                        end
                    end
                end
            end
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label_x8 = label_x8(:, :, :, order);
label_x4 = label_x4(:, :, :, order); 
label_x2 = label_x2(:, :, :, order); 

bicubic_x2 = bicubic_x2(:, :, :, order);
bicubic_x4 = bicubic_x4(:, :, :, order);
bicubic_x8 = bicubic_x8(:, :, :, order);

%% writing to HDF5
chunksz = 256;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)

    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs_x2 = label_x2(:,:,:,last_read+1:last_read+chunksz);
    batchlabs_x4 = label_x4(:,:,:,last_read+1:last_read+chunksz);
    batchlabs_x8 = label_x8(:,:,:,last_read+1:last_read+chunksz);
    
    batchbicubic_x2 = bicubic_x2(:,:,:,last_read+1:last_read+chunksz);
    batchbicubic_x4 = bicubic_x4(:,:,:,last_read+1:last_read+chunksz);
    batchbicubic_x8 = bicubic_x8(:,:,:,last_read+1:last_read+chunksz);
    
    startloc = struct('dat',[1,1,1,totalct+1], 'lab_x2', [1,1,1,totalct+1], 'lab_x4', [1,1,1,totalct+1], 'lab_x8', [1,1,1,totalct+1], 'bic_x2', [1,1,1,totalct+1], 'bic_x4', [1,1,1,totalct+1], 'bic_x8', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs_x2, batchlabs_x4, batchlabs_x8, batchbicubic_x2, batchbicubic_x4, batchbicubic_x8, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

toc