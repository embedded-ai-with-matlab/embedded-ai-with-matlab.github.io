function split = splitRecordings(class,counts)
%SPLITRECORDINGS Stratified train/validation/test split by recording.
%   split = splitRecordings(class,counts) assigns each recording to "train",
%   "val" or "test". Within each class, recordings are shuffled and the
%   first counts(1) go to training, the next counts(2) to validation, and
%   the next counts(3) to testing.
%
%   Splitting whole recordings, rather than individual windows, keeps
%   windows from the same 30 s recording out of different sets. Windows
%   from one recording share the board's orientation and the way it was
%   handled, so mixing them would make the test set too easy.

arguments
    class categorical
    counts (1,3) double = [9 3 3]
end

split = strings(numel(class),1);
for c = categories(class)'
    idx = find(class == c{1});
    idx = idx(randperm(numel(idx)));
    split(idx(1:counts(1))) = "train";
    split(idx(counts(1)+1:counts(1)+counts(2))) = "val";
    split(idx(counts(1)+counts(2)+1:sum(counts))) = "test";
end
end
