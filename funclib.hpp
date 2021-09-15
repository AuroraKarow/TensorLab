mtx::matrix vec_travel(mtx::matrix &vec_val, double (*func)(double&))
{
    auto vec_cpy = vec_val;
    vec_cpy.travel(func);
    return vec_cpy;
}

double sigmoid(double &val){
    return 1 / (1 + 1 / exp(val));
}

double sigmoid_dv(double &val)
{
    return sigmoid(val) * (1.0 - sigmoid(val));
}

mtx::matrix sigmoid(mtx::matrix &vec_val)
{
    return vec_travel(vec_val, sigmoid);
}

mtx::matrix sigmoid_dv(mtx::matrix &vec_val)
{
    return vec_travel(vec_val, sigmoid_dv);
}

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

mtx::matrix ReLU(mtx::matrix &vec_val)
{
    return vec_travel(vec_val, ReLU);
}

mtx::matrix ReLU_dv(mtx::matrix &vec_val)
{
    return vec_travel(vec_val, ReLU_dv);
}

mtx::matrix softmax(mtx::matrix &vec_val)
{
    mtx::matrix ans(vec_val.get_ln_cnt(), vec_val.get_col_cnt());
    double sum = 0;
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            sum += exp(vec_val[i][j]);
    for(auto i=0; i<vec_val.get_ln_cnt(); ++i)
        for(auto j=0; j<vec_val.get_col_cnt(); ++j)
            ans[i][j] = exp(vec_val[i][j]) / sum;
    return ans;
}

mtx::matrix softmax_grad(mtx::matrix &output, mtx::matrix &origin) {return output - origin;}