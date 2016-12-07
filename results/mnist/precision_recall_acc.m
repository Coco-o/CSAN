function [ gan_pr, gan_re, gan_acc, kmm_pr, kmm_re, kmm_acc ] = precision_recall_accuracy( labels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

suffix = '';
for i = 1 : length(labels) - 1
    suffix = strcat(suffix, num2str(labels(i)), '_');
end
suffix = strcat(suffix, num2str(labels(end)));

gan = reshape(csvread(strcat('gan_res', suffix, '.csv')), [500, 10]);
kmm = reshape(csvread(strcat('kmm_res', suffix, '.csv')), [500, 10]);

gan_accept = sum(gan >= 0.5);
kmm_accept = sum(kmm >= 1);

gan_pr = sum(gan_accept(labels + 1)) / sum(gan_accept);
kmm_pr = sum(kmm_accept(labels + 1)) / sum(kmm_accept);
gan_re = sum(gan_accept(labels + 1)) / (length(labels) * size(gan, 1));
kmm_re = sum(kmm_accept(labels + 1)) / (length(labels) * size(kmm, 1));

truth = zeros(500, 10);
truth(:, labels + 1) = 1;
gan_acc = sum(sum((gan >= 0.5) == truth)) / numel(gan);
kmm_acc = sum(sum((kmm >= 1) == truth)) / numel(kmm);

end

