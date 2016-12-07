function [ gan_pr, gan_re, kmm_pr, kmm_re ] = precision_recall( labels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

suffix = '';
for i = 1 : length(labels) - 1
    suffix = strcat(suffix, num2str(labels(i)), '_');
end
suffix = strcat(suffix, num2str(labels(end)));

gan = reshape(csvread(strcat('gan_res', suffix, '.csv')), [500, 10]);
kmm = reshape(csvread(strcat('kmm_res', suffix, '.csv')), [500, 10]);

gan_acc = sum(gan >= 0.5);
kmm_acc = sum(kmm >= 1);

gan_pr = sum(gan_acc(labels + 1)) / sum(gan_acc);
kmm_pr = sum(kmm_acc(labels + 1)) / sum(kmm_acc);
gan_re = sum(gan_acc(labels + 1)) / (length(labels) * size(gan, 1));
kmm_re = sum(kmm_acc(labels + 1)) / (length(labels) * size(kmm, 1));

end

