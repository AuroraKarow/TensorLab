vect vec_travel(vect &vec_val, double (*func)(double&))
{
    auto copy_vec = vec_val;
    for(auto i=0; i<vec_val.ELEM_CNT; ++i) copy_vec.pos_idx(i) = func(vec_val.pos_idx(i));
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