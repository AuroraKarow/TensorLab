vect vec_travel(vect &vec_val, double (*func)(double&))
{
    auto copy_vec = vec_val;
    for(auto i=0; i<vec_val.ELEM_CNT; ++i) copy_vec.pos_idx(i) = func(vec_val.pos_idx(i));
    return copy_vec;
}

double sigmoid(double &val){ return 1 / (1 + 1 / exp(val)); }

vect sigmoid(vect &vec_val){return vec_travel(vec_val, sigmoid);}

template<typename T> set<T> sigmoid(set<T> &set_val)
{
    set<T> ans(set_val.size());
    for(auto i=0; i<set_val.size(); ++i) ans[i] = sigmoid(set_val[i]);
    return ans;
}

double sigmoid_dv(double &val){ return sigmoid(val) * (1.0 - sigmoid(val)); }

vect sigmoid_dv(vect &vec_val){return vec_travel(vec_val, sigmoid_dv);}

template<typename T> set<T> sigmoid_dv(set<T> &set_val)
{
    set<T> ans(set_val.size());
    for(auto i=0; i<set_val.size(); ++i) ans[i] = sigmoid_dv(set_val[i]);
    return ans;
}

double ReLU(double &val)
{
    if(val < 0) return 0;
    else return val;
}

vect ReLU(vect &vec_val){return vec_travel(vec_val, ReLU);}

template<typename T> set<T> ReLU(set<T> &set_val)
{
    set<T> ans(set_val.size());
    for(auto i=0; i<set_val.size(); ++i) ans[i] = ReLU(set_val[i]);
    return ans;
}

double ReLU_dv(double &val)
{
    if(val < 0) return 0;
    else return 1;
}

vect ReLU_dv(vect &vec_val){return vec_travel(vec_val, ReLU_dv);}

template<typename T> set<T> ReLU_dv(set<T> &set_val)
{
    set<T> ans(set_val.size());
    for(auto i=0; i<set_val.size(); ++i) ans[i] = ReLU_dv(set_val[i]);
    return ans;
}

vect softmax(vect &vec_val)
{
    vect ans(vec_val.get_ln_cnt(), vec_val.get_col_cnt());
    double sum = 0;
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            sum += std::exp(vec_val[i][j]);
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            ans[i][j] = std::exp(vec_val[i][j]) / sum;
    return ans;
}

template<typename T> set<T> softmax(set<T> &set_vec)
{
    set<T> ans(set_vec.size());
    for(auto i=0; i<ans.size(); ++i) ans[i] = softmax(set_vec[i]);
    return ans;
}

vect softmax_dv(vect &vec_input, vect &vec_output)
{
    auto elem_cnt = vec_input.ELEM_CNT;
    if(vec_input.shape_valid(vec_output) && vec_input.LN_CNT==elem_cnt)
    {
        vect ans(elem_cnt, 1);
        for(auto i=0; i<elem_cnt; ++i) for(auto j=0; j<elem_cnt; ++j)
            if(i==j) ans.pos_idx(j) += vec_output.pos_idx(i) * (1 - vec_output.pos_idx(j));
            else ans.pos_idx(j) += (-1) * vec_output.pos_idx(j) * vec_output.pos_idx(i);
        return ans;
    }
    else return blank_vect;
}

template<typename T> set<T> softmax_dv(set<T> &set_vec)
{
    set<T> ans(set_vec.size());
    for(auto i=0; i<ans.size(); ++i) ans[i] = softmax(set_vec[i]);
    return ans;
}

vect cec_grad(vect &output, vect &origin)
{
    auto elem_cnt = output.ELEM_CNT;
    vect ans(elem_cnt, 1);
    if(output.shape_valid(origin) && elem_cnt==origin.LN_CNT)
    {
        auto orgn_sum = origin.elem_sum();
        for(auto i=0; i<elem_cnt; ++i)
            ans.pos_idx(i) = (-1) * orgn_sum / output.pos_idx(i);
    } 
    return ans;
}

feature cec_grad(feature &output, feature &origin)
{
    if(output.size() == origin.size())
    {
        feature ans(output.size());
        for(auto i=0; i<origin.size(); ++i) ans[i] = cec_grad(output[i], origin[i]);
        return ans;
    }
    else return blank_feature;
}
set<feature> cec_grad(set<feature> &output, set<feature> &origin)
{
    if(output.size() == origin.size())
    {
        set<feature> ans(output.size());
        for(auto i=0; i<origin.size(); ++i) ans[i] = cec_grad(output[i], origin[i]);
        return ans;
    }
    else return output;
}

vect softmax_cec_grad(vect &softmax_output, vect &origin) {return softmax_output - origin;}

template<typename T> set<T> softmax_cec_grad(set<T> &setSoftmaxOutput, set<T> &setOrigin)
{
    if(setSoftmaxOutput.size() == setOrigin.size())
    {
        set<T> setGradOutput(setOrigin.size());
        for(auto i=0; i<setOrigin.size(); ++i) setGradOutput[i] = softmax_cec_grad(setSoftmaxOutput[i], setOrigin[i]);
        return setGradOutput;
    }
    else return set<T>::blank_queue();
}

vect hadamard_produc(vect &l_vec, vect &r_vec) { return l_vec.elem_cal_opt(r_vec, MATRIX_ELEM_MULT); }

template<typename T> set<T> hadamard_produc(set<T> &l_set, set<T> &r_set)
{
    set<T> ans(l_set.size());
    for(auto i=0; i<ans.size(); ++i) ans[i] = hadamard_produc(l_set[i], r_set[i]);
    return ans;
}

vect divisor_dominate(vect &divisor, double epsilon)
{
    auto cpy_val = divisor;
    for(auto i=0; i<cpy_val.get_ln_cnt(); ++i)
        for(auto j=0; j<cpy_val.get_col_cnt(); ++j)
            if(cpy_val[i][j] == 0) cpy_val[i][j] = epsilon;
    return cpy_val;
}

uint64_t samp_block_cnt(uint64_t filter_dir_cnt, uint64_t dir_dilation) {return (dir_dilation + 1) * filter_dir_cnt - dir_dilation;}

uint64_t samp_trace_pos(uint64_t output_dir_pos, uint64_t filter_dir_pos, uint64_t dir_stride, uint64_t dir_dilation) {return output_dir_pos * dir_stride + filter_dir_pos * (1 + dir_dilation);}

uint64_t samp_output_dir_cnt(uint64_t input_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (input_dir_cnt - samp_block_cnt(filter_dir_cnt, dir_dilation)) / dir_stride + 1;}

uint64_t samp_input_dir_cnt(uint64_t output_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (output_dir_cnt - 1) * dir_stride + samp_block_cnt(filter_dir_cnt, dir_dilation);}

bool samp_valid(uint64_t input_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (input_dir_cnt - samp_block_cnt(filter_dir_cnt, dir_dilation)) % dir_stride == 0;}

feature merge_channel(tensor &input)
{
    feature ft_map(input.size());
    for(auto i=0; i<input.size(); ++i) for(auto j=0; j<input[i].size(); ++j)
    {
        if(ft_map[i].is_matrix()) ft_map[i] += input[i][j];
        else ft_map[i] = input[i][j];
        if(!ft_map[i].is_matrix()) return blank_feature;
    }
    return ft_map;
}

set<feature> merge_channel(set<tensor> &input)
{
    set<feature> set_ft_map(input.size());
    for(auto i=0; i<input.size(); ++i) for(auto j=0; j<input[i].size(); ++j)
    {
        set_ft_map[i] = merge_channel(input[i]);
        if(!set_ft_map[i].size()) return blank_ft_seq;
    }
    return set_ft_map;
}