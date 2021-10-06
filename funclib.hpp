vect vec_travel(vect &vec_val, double (*func)(double&))
{
    auto copy_vec = vec_val;
    for(auto i=0; i<vec_val.ELEM_CNT; ++i) copy_vec.pos_idx(i) = func(vec_val.pos_idx(i));
    return copy_vec;
}

double sigmoid(double &val){ return 1 / (1 + 1 / exp(val)); }

double sigmoid_dv(double &val){ return sigmoid(val) * (1.0 - sigmoid(val)); }

vect sigmoid(vect &vec_val){return vec_travel(vec_val, sigmoid);}

vect sigmoid_dv(vect &vec_val){return vec_travel(vec_val, sigmoid_dv);}

double ReLU(double &val)
{
    if(val < 0) return 0;
    else return val;
}

double ReLU_dv(double &val)
{
    if(val < 0) return 0;
    else return 1;
}

vect ReLU(vect &vec_val){return vec_travel(vec_val, ReLU);}

vect ReLU_dv(vect &vec_val){return vec_travel(vec_val, ReLU_dv);}

vect softmax(vect &vec_val)
{
    vect ans(vec_val.get_ln_cnt(), vec_val.get_col_cnt());
    double sum = 0;
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            sum += std::exp(vec_val[i][j]);
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            ans[i][j] = exp(vec_val[i][j]) / sum;
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

vect softmax_cec_grad(vect &softmax_output, vect &origin) {return softmax_output - origin;}

vect divisor_dominate(vect &divisor, double epsilon)
{
    auto cpy_val = divisor;
    for(auto i=0; i<cpy_val.get_ln_cnt(); ++i)
        for(auto j=0; j<cpy_val.get_col_cnt(); ++j)
            if(cpy_val[i][j] == 0) cpy_val[i][j] = epsilon;
    return cpy_val;
}

feature activate(feature &input, vect(*act_func)(vect&))
{
    feature output(input.size());
    for(auto i=0; i<input.size(); ++i) output[i] = act_func(input[i]);
    return output;
}

set<feature> activate(set<feature> &input, vect(*act_func)(vect&))
{
    set<feature> output(input.size());
    for(auto i=0; i<input.size(); ++i) output[i] = activate(input[i], act_func);
    return output;
}

feature derivative(feature &act_input, feature &grad, vect(*act_func_dv)(vect&))
{
    feature grad_dv(act_input.size());
    for(auto i=0; i<act_input.size(); ++i)
    {
        grad_dv[i] = act_func_dv(act_input[i]).elem_cal_opt(grad[i], MATRIX_ELEM_MULT);
        if(!grad_dv[i].is_matrix()) return blank_feature;
    }
    return grad_dv;
}
set<feature> derivative(set<feature> &act_input, set<feature> &grad, vect(*act_func_dv)(vect&))
{
    set<feature> grad_dv(act_input.size());
    for(auto i=0; i<act_input.size(); ++i)
    {
        grad_dv[i] = derivative(act_input[i], grad[i], act_func_dv);
        if(!grad_dv[i].size()) return blank_ft_seq;
    }
    return grad_dv;
}

uint64_t samp_block_cnt(uint64_t filter_dir_cnt, uint64_t dir_dilation) {return (dir_dilation + 1) * filter_dir_cnt - dir_dilation;}

uint64_t samp_trace_pos(uint64_t output_dir_pos, uint64_t filter_dir_pos, uint64_t dir_stride, uint64_t dir_dilation) {return output_dir_pos * dir_stride + filter_dir_pos * (1 + dir_dilation);}

uint64_t samp_output_dir_cnt(uint64_t input_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return input_dir_cnt - samp_block_cnt(filter_dir_cnt, dir_stride) / dir_dilation + 1;}

uint64_t samp_input_dir_cnt(uint64_t output_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (output_dir_cnt - 1) * dir_stride + samp_block_cnt(filter_dir_cnt, dir_dilation);}

bool samp_valid(uint64_t input_dir_cnt, uint64_t filter_dir_cnt, uint64_t dir_stride, uint64_t dir_dilation) {return (input_dir_cnt - samp_block_cnt(filter_dir_cnt, dir_dilation)) % dir_stride == 0;}

feature merge_channel(tensor &input)
{
    feature ft_map(input.size());
    for(auto i=0; i<input.size(); ++i)
    {
        vect sgl_chann;
        for(auto j=0; j<input[i].size(); ++j)
        {
            if(sgl_chann.is_matrix()) sgl_chann += input[i][j];
            else sgl_chann = input[i][j];
            if(!sgl_chann.is_matrix()) return blank_feature;
        }
        ft_map[i] = sgl_chann;
    }
    return ft_map;
}

set<feature> merge_channel(set<tensor> &input)
{
    set<feature> set_ft_map(input.size());
    for(auto i=0; i<input.size(); ++i)
    {
        feature sgl_ft;
        for(auto j=0; j<input[i].size(); ++j)
        {
            set_ft_map[i] = merge_channel(input[i]);
            if(!sgl_ft.size()) return blank_ft_seq;
        }
    }
    return set_ft_map;
}