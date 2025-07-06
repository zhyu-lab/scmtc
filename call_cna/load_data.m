function [data_lrc_all,data_chr_all,data_bin_all,bin_size] = load_data(inputFile,embeddingFile,K)

data_embedding_all = load(embeddingFile);
num_cell = size(data_embedding_all,1);
disp(['The size of embedding: [' num2str(size(data_embedding_all,1)) ', ' num2str(size(data_embedding_all,2)) ']']);

% Compute pairwise distances
D = pdist2(data_embedding_all,data_embedding_all);

% Find k nearest neighbors for each cell
[~, idx] = sort(D,2,'ascend');
neighbors = idx(:,1:K);

% Build adjacency matrix (symmetric kNN graph)
knn_graph = zeros(num_cell);
for i = 1:num_cell
    for j = 1:K
        knn_graph(i,neighbors(i,j)) = 1;
        knn_graph(neighbors(i,j),i) = 1;  % make the graph undirected
    end
end

fid = fopen(inputFile, 'r');
line = fgetl(fid);
fields = regexp(line,'=','split');
bin_size = str2double(strtrim(fields{2}));
line = fgetl(fid);
fields = regexp(line,',','split');
cell_labels = fields(4:end);
results = textscan(fid,repmat('%f',1,length(cell_labels)+3),'delimiter',',');
data_chr_all = results{1};
data_gc_all = results{2};
data_map_all = results{3};
data_rc_all = cell2mat(results(4:end));
clear results;
fclose(fid);

chromosomes = unique(data_chr_all);
data_bin_all = zeros(length(data_chr_all),1);
for i = 1:length(chromosomes)
    tv = data_chr_all == chromosomes(i);
    data_bin_all(tv) = 1:sum(tv);
end

% filter by GC-content
tv = data_gc_all > 0.1 & data_gc_all < 0.9;
data_chr_all = data_chr_all(tv);
data_rc_all = data_rc_all(tv,:);
data_bin_all = data_bin_all(tv);

% KNN based smoothing
data_rc_all_n = zeros(size(data_rc_all));
for k = 1:num_cell
    tv = knn_graph(k,:) == 1;
    data_rc_all_n(:,k) = sum(data_rc_all(:,tv),2);
end
data_rc_all = data_rc_all_n;

% median normalization
m_rc_cells = median(data_rc_all,1);
data_rc_all = data_rc_all./(repmat(m_rc_cells,size(data_rc_all,1),1)+eps);

% filter bins by read counts
mean_rc_bins = mean(data_rc_all,2);
tv = mean_rc_bins > prctile(mean_rc_bins,1) & mean_rc_bins < prctile(mean_rc_bins,99);
data_chr_all = data_chr_all(tv);
data_rc_all = data_rc_all(tv,:);
data_bin_all = data_bin_all(tv);

data_lrc_all = log2(data_rc_all'+eps);
disp(['After data filtering, the size of data becomes to [' num2str(size(data_lrc_all,1)) ', ' num2str(size(data_lrc_all,2)) ']']);

end