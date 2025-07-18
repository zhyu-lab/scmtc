function segments = segment_results(state_seq)


pre_state = [];
s_indx = [];
segments = [];
for i = 1:length(state_seq)
    if isempty(pre_state)
        pre_state = state_seq(i);
        s_indx = i;
    elseif state_seq(i) ~= pre_state
        segments = [segments;s_indx i-1 pre_state];
        pre_state = state_seq(i);
        s_indx = i;
    end
end

if s_indx <= length(state_seq)
    segments = [segments;s_indx length(state_seq) pre_state];
end

end