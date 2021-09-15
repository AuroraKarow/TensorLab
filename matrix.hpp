MATRIX_BEGIN

bool mtx_pos_valid(uint64_t ln, uint64_t col, uint64_t ln_cnt, uint64_t col_cnt)
{
    if (ln<ln_cnt && col<col_cnt) return true;
    else return false;
}

std::ostream &operator<<(std::ostream &output, const mtx_pos &pos)
{
    output << '(' << pos.ln << ", " << pos.col << ')';
    return output;
}

mtx_pos mtx_elem_pos(uint64_t idx, uint64_t ln_cnt, uint64_t col_cnt)
{
    mtx_pos pos;
    if(ln_cnt && col_cnt)
    {
        pos.ln = idx / ln_cnt;
        pos.col = idx % col_cnt;
    }
    return pos;
}
uint64_t mtx_elem_pos(uint64_t ln, uint64_t col, uint64_t ln_cnt, uint64_t col_cnt)
{
    if (mtx_pos_valid(ln, col, ln_cnt, col_cnt)) return ln * col_cnt + col;
    else return 0;
}

MATRIX mtx_init(uint64_t elem_cnt)
{
    auto mtx_ptr = std::make_unique<double[]>(elem_cnt);
    for(auto i=0; i<elem_cnt; ++i) mtx_ptr[i] = 0;
    return mtx_ptr;
}
MATRIX mtx_init(uint64_t ln_cnt, uint64_t col_cnt) {return mtx_init(ln_cnt * col_cnt);}

MATRIX mtx_init_E(uint64_t dms)
{
    auto elem_cnt = dms * dms;
    auto mtx_ptr = mtx_init(elem_cnt);
    for(auto i=0; i<elem_cnt; ++i) mtx_ptr[i*(dms+1)] = 1;
    return mtx_ptr;
}

void mtx_abs(MATRIX &mtx_src, uint64_t elem_cnt) {if(mtx_src && elem_cnt) for(auto i=0; i<elem_cnt; ++i) mtx_src[i] = std::abs(mtx_src[i]);}

MATRIX mtx_init_rand(uint64_t elem_cnt, double boundry_first = 0, double boundry_second = 0, double acc = 1e-5)
{
    MATRIX mtx_val = mtx_init(elem_cnt);
    for (int i = 0; i < elem_cnt; ++i)
        if(boundry_first==boundry_second) mtx_val[i] = (((double)rand_e() / (double)rand_e._Max) - 0.5) * 2.0;
        else mtx_val[i] = bagrt::random_number(boundry_first, boundry_second, false, acc); 
    return mtx_val;
}

mtx_info mtx_child_vec(MATRIX &mtx_src, uint64_t from_ln, uint64_t to_ln, uint64_t from_col, uint64_t to_col, uint64_t ln_cnt, uint64_t col_cnt, uint64_t ln_dilation = 0, uint64_t col_dilation = 1)
{
    mtx_info mtx_info;
    if(mtx_src && col_cnt && ln_cnt &&
        from_ln>=0 && to_ln>=from_ln && ln_cnt>to_ln &&
        from_col>=0 && to_col>=from_col && col_cnt>to_col)
    {
        mtx_info.ln_cnt = (bagrt::num_cnt(from_ln, to_ln) + ln_dilation) / (ln_dilation + 1);
        mtx_info.col_cnt = (bagrt::num_cnt(from_col, to_col) + col_dilation) / (col_dilation + 1);
        mtx_info.mtx_val = mtx_init(mtx_info.ln_cnt, mtx_info.col_cnt);
        for(auto i=0; i<mtx_info.ln_cnt; ++i)
            for(auto j=0; j<mtx_info.col_cnt; ++j)
            {
                auto curr_orgn_ln = from_ln + i * (1 + ln_dilation);
                auto curr_orgn_col = from_col + j * (1 + col_dilation);
                mtx_info.mtx_val[mtx_elem_pos(i, j, mtx_info.ln_cnt, mtx_info.col_cnt)] = mtx_src[mtx_elem_pos(curr_orgn_ln, curr_orgn_col, ln_cnt, col_cnt)];
            }
    }
    return mtx_info;
}

double mtx_atom(MATRIX &mtx_val) {return mtx_val[0];}

bool mtx_refresh(MATRIX &mtx_val, uint64_t elem_cnt)
{
    if(mtx_val)
    {
        for(auto i=0; i<elem_cnt; ++i)
            if(mtx_val[i]) mtx_val[i] = 0;
        return true;
    }
    else return false;
}
bool mtx_refresh(MATRIX &mtx_val, uint64_t ln_cnt, uint64_t col_cnt) {return mtx_refresh(mtx_val, ln_cnt * col_cnt);}

void mtx_show(MATRIX &mtx_val, uint64_t ln_cnt, uint64_t col_cnt)
{
    if(mtx_val)
    {
        auto elem_amt = ln_cnt * col_cnt;
        for (auto i = 0; i < elem_amt; ++i)
        {
            std::cout << mtx_val[i];
            if ((i + 1) % col_cnt) std::cout << '\t';
            else std::cout << std::endl;
        }
    }
}

double mtx_det(MATRIX &mtx_val, int dms)
{
    if(mtx_val)
    {
        if (dms == 1)
        return mtx_val[0];
        auto ac = mtx_init((dms - 1) * (dms - 1));
        int mov = 0;
        double sum = 0.0;
        for (int arow = 0; arow < dms; arow++)
        {
            for (int brow = 0; brow < dms - 1; brow++)
            {
                mov = arow > brow ? 0 : 1;
                for (int j = 0; j < dms - 1; ++j)
                    ac[brow * (dms - 1) + j] = mtx_val[(brow + mov) * dms + j + 1];
            }
            int flag = (arow % 2 == 0 ? 1 : -1);
            sum += flag * mtx_val[arow * dms] * mtx_det(ac, dms - 1);
        }
        return sum;
    }
    else return -1;    
}

mtx_extm mtx_extm_val(MATRIX &mtx_val, uint64_t from_ln, uint64_t to_ln, uint64_t from_col, uint64_t to_col, uint64_t ln_cnt, uint64_t col_cnt, uint64_t ln_dilation = 0, uint64_t col_dilation = 0, bool max_flag = true)
{
    mtx_extm mtx_info;
    auto ln_block = ln_dilation + 1,
        col_block = col_dilation + 1,
        ln_elem_block = bagrt::num_cnt(from_ln, to_ln),
        col_elem_block = bagrt::num_cnt(from_col, to_col);
    if(mtx_val && col_cnt && ln_cnt &&
        from_ln>=0 && to_ln>=from_ln && ln_cnt>to_ln &&
        from_col>=0 && to_col>=from_col && col_cnt>to_col &&
        (ln_elem_block-1)%ln_block==0 && (col_elem_block-1)%col_block==0)
    {
        mtx_info.val = mtx_val[mtx_elem_pos(from_ln, from_col, ln_cnt, col_cnt)];
        for(auto i=from_ln; i<=to_ln; ++(i+=ln_dilation))
            for(auto j=from_col; j<=to_col; ++(j+=col_dilation))
            {
                auto curr_no = mtx_elem_pos(i, j, ln_cnt, col_cnt);
                mtx_pos curr_pos;
                curr_pos.ln = i;
                curr_pos.col = j;
                if((max_flag&&mtx_val[curr_no]>mtx_info.val) || (!max_flag&&mtx_val[curr_no]<mtx_info.val))
                {
                    mtx_info.val = mtx_val[curr_no];
                    mtx_info.pos_list.reset();
                    mtx_info.pos_list.emplace_back(curr_pos);
                }
                else if(mtx_val[curr_no] == mtx_info.val) mtx_info.pos_list.emplace_back(curr_pos);
                else continue;
            }
    }
    return mtx_info;
}

double mtx_sum(MATRIX &mtx_val, uint64_t from_ln, uint64_t to_ln, uint64_t from_col, uint64_t to_col, uint64_t ln_cnt, uint64_t col_cnt, uint64_t ln_dilation = 0, uint64_t col_dilation = 0)
{
    auto sum = 0.0;
    if(mtx_val && col_cnt && ln_cnt &&
        from_ln>=0 && to_ln>=from_ln && ln_cnt>to_ln &&
        from_col>=0 && to_col>=from_col && col_cnt>to_col)
    {
        auto mtx_ch = mtx_child_vec(mtx_val, from_ln, to_ln, from_col, to_col, ln_cnt, col_cnt, ln_dilation, col_dilation);
        auto elem_cnt = mtx_ch.col_cnt * mtx_ch.ln_cnt;
        for(auto i=0; i<elem_cnt; ++i) sum += mtx_ch.mtx_val[i];
    }
    return sum;
}

MATRIX mtx_copy(MATRIX &mtx_src, uint64_t elem_cnt)
{
    auto mtx_val = mtx_init(elem_cnt);
    if(mtx_src) for(auto i=0; i<elem_cnt; ++i) mtx_val[i] = mtx_src[i];
    return mtx_val;
}
MATRIX mtx_copy(MATRIX &mtx_src, uint64_t ln_cnt, uint64_t col_cnt) {return mtx_copy(mtx_src, ln_cnt * col_cnt);}

MATRIX mtx_from_str(std::string &mtx_str, uint64_t &line, uint64_t &column)
{
    column = 0;
    line = 0;
    int cnt = 0;
    // get column amount
    while (mtx_str[cnt] != '\n' && mtx_str[cnt] != '\0')
    {
        if (mtx_str[cnt] == '.' || mtx_str[cnt] == '-' || mtx_str[cnt] == '+' ||
            (mtx_str[cnt] >= '0' && mtx_str[cnt] <= '9') || mtx_str[cnt] == '/')
            if (mtx_str[cnt - 1] == ' ' || mtx_str[cnt - 1] == '\t' || cnt - 1 < 0)
                column++;
        cnt++;
    }
    for (int i = 0; i < mtx_str.length(); ++i) if (mtx_str[i] == '\n') line++;
    return bagrt::extract_number(mtx_str);
}

MATRIX mtx_transposition(MATRIX &mtx_src, uint64_t ln_cnt, uint64_t col_cnt)
{
    if(mtx_src && ln_cnt && col_cnt)
    {
        auto elem_cnt = ln_cnt * col_cnt;
        auto mtx_trans = mtx_init(elem_cnt);
        for (auto i = 0; i < ln_cnt; ++i)
            for (auto j = 0; j < col_cnt; ++j)
                mtx_trans[mtx_elem_pos(j, i, col_cnt, ln_cnt)] = mtx_src[mtx_elem_pos(i, j, ln_cnt, col_cnt)];
        return mtx_trans;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_add(MATRIX &l_mtx, MATRIX &r_mtx, uint64_t ln_cnt, uint64_t col_cnt, bool nega = false)
{
    if(l_mtx && r_mtx && ln_cnt && col_cnt)
    {
        auto elem_cnt = ln_cnt * col_cnt;
        auto mtx_val = mtx_copy(l_mtx, elem_cnt);
        for(auto i=0; i<elem_cnt; ++i)
            if(nega) mtx_val[i] += r_mtx[i] * (-1);
            else mtx_val[i] += r_mtx[i];
        return mtx_val;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_mult(MATRIX &l_mtx, MATRIX &r_mtx, uint64_t l_ln_cnt, uint64_t l_col_cnt, uint64_t r_ln_cnt, uint64_t r_col_cnt)
{
    if(l_mtx && r_mtx && l_ln_cnt && l_col_cnt && r_ln_cnt && r_col_cnt && l_col_cnt==r_ln_cnt)
    {
        auto res_mtx = mtx_init(l_ln_cnt, r_col_cnt);
        for (int i = 0; i < l_ln_cnt; ++i)
            for (int j = 0; j < r_col_cnt; ++j)
            {
                double sum = 0.0;
                for (int k = 0; k < l_col_cnt; ++k)
                    sum += l_mtx[mtx_elem_pos(i, k, l_ln_cnt, l_col_cnt)] * r_mtx[mtx_elem_pos(k, j, r_ln_cnt, r_col_cnt)];
                res_mtx[mtx_elem_pos(i, j, l_ln_cnt, r_col_cnt)] = sum;
            }
        return res_mtx;
    }
    else return MATRIX_NULL;
}
MATRIX mtx_mult(MATRIX &mtx_src, double val, uint64_t ln_cnt, uint64_t col_cnt)
{
    if(mtx_src && ln_cnt && col_cnt)
    {
        auto elem_cnt = ln_cnt * col_cnt;
        auto res_mtx = mtx_copy(mtx_src, elem_cnt);
        for(auto i=0; i<elem_cnt; ++i) res_mtx[i] *= val;
        return res_mtx;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_elem_cal_opt(MATRIX &l_mtx, MATRIX &r_mtx, uint64_t ln_cnt, uint64_t col_cnt, uint64_t opt_idx)
{
    if(l_mtx && r_mtx && ln_cnt && col_cnt)
    {
        auto elem_cnt = ln_cnt * col_cnt;
        MATRIX mtx_val;
        for(auto i=0; i<elem_cnt; ++i)
            switch (opt_idx)
            {
            case ELEM_MULT:
                mtx_val = mtx_init(elem_cnt);
                mtx_val[i] = l_mtx[i] * r_mtx[i];
                break;
            case ELEM_DIV:
                mtx_val = mtx_init(elem_cnt);
                if(r_mtx[i])
                {
                    mtx_val[i] = l_mtx[i] / r_mtx[i];
                    break;
                }
            default: return MATRIX_NULL;
            }
        return mtx_val;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_elem_cal_opt(MATRIX &mtx_val, double para, uint64_t ln_cnt, uint64_t col_cnt, uint64_t opt_idx)
{
    if(mtx_val && ln_cnt && col_cnt)
    {
        auto elem_cnt = ln_cnt * col_cnt;
        auto mtx_res = mtx_init(elem_cnt);
        for(auto i=0; i<elem_cnt; ++i)
            switch (opt_idx)
            {
            case ELEM_POW:
                mtx_res[i] = std::pow(mtx_val[i], para);
                break;
            case ELEM_DIV:
                if(para)
                {
                    mtx_res[i] = mtx_val[i] / para;
                    break;
                }
            default: return MATRIX_NULL;
            }
        return mtx_res;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_broadcast_add(MATRIX &mtx_val, uint64_t elem_cnt, double val)
{
    if(mtx_val && elem_cnt)
    {
        auto mtx_res = mtx_copy(mtx_val, elem_cnt);
        for(auto i=0; i<elem_cnt; ++i) mtx_res[i] += val;
        return mtx_res;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_broadcast_subtract(MATRIX &mtx_val, uint64_t elem_cnt, double val, bool val_subtrahend = true)
{
    if(mtx_val && elem_cnt)
    {
        auto mtx_res = mtx_copy(mtx_val, elem_cnt);
        for(auto i=0; i<elem_cnt; ++i)
            if(val_subtrahend) mtx_res[i] -= val;
            else mtx_res[i] = val - mtx_res[i];
        return mtx_res;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_LU(MATRIX &mtx_val, int dms)
{
    if(mtx_val && dms)
    {
        auto lu = mtx_copy(mtx_val, dms, dms);
        auto e = mtx_init_E(dms);
        auto l = mtx_copy(e, dms, dms);
        for (int i = 0; i < dms - 1; ++i)
        {
            auto tool = mtx_copy(e, dms, dms);
            for (int j = dms - 1; j > i; --j) tool[mtx_elem_pos(j, i, dms, dms)] = (-1.0) * lu[mtx_elem_pos(j, i, dms, dms)] / lu[mtx_elem_pos(i, i, dms, dms)];
            for (int j = dms - 1; j > i; --j) l[mtx_elem_pos(j, i, dms, dms)] = (-1.0) * tool[mtx_elem_pos(j, i, dms, dms)];
            for (int j = dms - 1; j > i; --j) lu[mtx_elem_pos(j, i, dms, dms)] = (-1.0) * tool[mtx_elem_pos(j, i, dms, dms)];
            lu = mtx_mult(tool, lu, dms, dms, dms, dms);
        }
        // get the inversion matrix
        for (int i = 0; i < dms; ++i)
            for (int j = 0; j < dms; ++j)
                if (i > j) lu[mtx_elem_pos(i, j, dms, dms)] = l[mtx_elem_pos(i, j, dms, dms)];
        return lu;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_equation(MATRIX &coefficient, MATRIX &b, int dms)
{
    if(coefficient && b && dms)
    {
        auto lu = mtx_LU(coefficient, dms);
        auto e = mtx_init_E(dms);
        auto l = mtx_copy(e, dms, dms);
        auto u = mtx_copy(e, dms, dms);
        for (int i = 0; i < dms; ++i)
            for (int j = 0; j < dms; ++j)
                if (i <= j) u[mtx_elem_pos(i, j, dms, dms)] = lu[mtx_elem_pos(i, j, dms, dms)];
                else l[mtx_elem_pos(i, j, dms, dms)] = lu[mtx_elem_pos(i, j, dms, dms)];
        auto y = mtx_init(dms), x = mtx_init(dms);
        for (int i = 0; i < dms; ++i) if (i)
        {
            double temp = 0.0;
            for (int j=0; j < i; ++j) temp += y[j] * l[mtx_elem_pos(i, j, dms, dms)] * 1.0;
            y[i] = b[i] - temp;
        }
        else y[i] = b[i];
        for (int i=dms; i>0; --i) if (i - dms)
        {
            double temp = 0.0;
            for (int n=i-1; n<dms-1; n++) temp += u[mtx_elem_pos(i-1, n+1, dms, dms)] * x[n+1];
            x[i-1] = (y[i-1] - temp) / u[mtx_elem_pos(i-1, i-1, dms, dms)];
        }
        else x[dms-1] = y[dms-1] / u[mtx_elem_pos(dms-1, dms-1, dms, dms)];
        return x;
    }
    else return MATRIX_NULL;
}

MATRIX mtx_inverser(MATRIX &mat_val, int dms)
{
    if(mat_val && dms)
    {
        auto e = mtx_init_E(dms);
        auto inverse = mtx_init(dms, dms);
        for (int i=0; i<dms; ++i)
        {
            auto b = mtx_init(dms);
            for (int j=0; j<dms; ++j) b[j] = e[mtx_elem_pos(j, i, dms, dms)];
            auto x = mtx_equation(mat_val, b, dms);
            for (int n=0; n<dms; ++n) inverse[mtx_elem_pos(n, i, dms, dms)] = x[n];
        }
        return inverse;
    } return MATRIX_NULL;
}

double mtx_max_eigenvalue(MATRIX &mtx_val, int dms, MATRIX &w, double error = 1e-5, double init_elem = 1)
{
    if(mtx_val && dms)
    {
        double lambda = 0.0, lambda_temp = 0.0;
        auto x = mtx_init(dms), w = mtx_init(dms);
        for (int i=0; i<dms; ++i) x[i] = init_elem;
        do
        {
            lambda = lambda_temp;
            auto temp = mtx_init(dms);
            for (int i=0; i<dms; ++i) temp[i] = x[i];
            auto temp_abs =  mtx_copy(temp, dms);
            mtx_abs(temp_abs, dms);
            double max = mtx_extm_val(temp_abs, 0, dms-1, 0, 0, dms, 1).val;
            for (int i=0; i<dms; ++i) x[i] /= max;
            for (int i=0; i<dms; ++i) w[i] = x[i];
            double sum = 0.0;
            for (int i=0; i<dms; ++i) sum += w[i];
            for (int i=0; i<dms; ++i) w[i] /= sum;
            x = mtx_mult(mtx_val, x, dms, dms, dms, 1);
            for (int i=0; i<dms; ++i) temp[i] = x[i];
            lambda_temp = mtx_extm_val(temp, 0, dms-1, 0, 0, dms, 1).val;
        } while (std::abs(lambda_temp - lambda) > error);
        return lambda;
    }
    else return -1;
}

MATRIX mtx_jacobi_iterate(MATRIX &coefficient, MATRIX &b, int dms, double error = 1e-5, double init_elem = 1)
{
    auto x = mtx_init(dms), temp = mtx_init(dms);
    for (int i=0; i<dms; ++i) temp[i] = init_elem;
    bool flag = false;
    do
    {
        flag = false;
        for (int i=0; i<dms; ++i) x[i] = temp[i];
        for (int i=0; i<dms; ++i)
        {
            double sum = 0.0;
            for (int j=0; j<dms; ++j) if (i != j) sum += coefficient[mtx_elem_pos(i, j, dms, dms)] * x[j];
            temp[i] = (b[i] - sum) / coefficient[mtx_elem_pos(i, i, dms, dms)];
        }
        for (int i=0; i<dms; ++i) if (std::abs(x[i] - temp[i]) >= error)
        {
            flag = true;
            break;
        }
    } while (flag);
    return x;
}

MATRIX mtx_rotate_rect(MATRIX &mtx_val, int ln_cnt, int col_cnt)
{
    auto rot_rec_mtx = mtx_init(ln_cnt, col_cnt);
    for (auto i = 0; i < col_cnt; ++i)
        for (auto j = 0; j < ln_cnt; ++j)
            rot_rec_mtx[mtx_elem_pos(i, j, ln_cnt, col_cnt)] = mtx_val[mtx_elem_pos(j, col_cnt - i - 1, ln_cnt, col_cnt)];
    return rot_rec_mtx;
}

MATRIX mtx_mirror_flip(MATRIX &mtx_src, uint64_t ln_cnt, uint64_t col_cnt)
{
    auto mtx_val = mtx_copy(mtx_src, ln_cnt, col_cnt);
    for(auto i=0; i<ln_cnt; ++i)
        for(int j=0,k=col_cnt-1; k>j; ++j,--k)
            std::swap(mtx_val[mtx_elem_pos(i,j,ln_cnt,col_cnt)], mtx_val[mtx_elem_pos(i,k,ln_cnt,col_cnt)]);
    return mtx_val;
}

mtx_info mtx_pad(MATRIX &mtx_val, uint64_t ln_cnt, uint64_t col_cnt, uint64_t ln_t, uint64_t col_r, uint64_t ln_b, uint64_t col_l, uint64_t ln_dist = 0, uint64_t col_dist = 0)
{
    mtx_info mtx_res;
    if(mtx_val && ln_cnt && col_cnt)
    {
        mtx_res.ln_cnt = ln_t + ln_b + ln_cnt + (ln_cnt - 1) * ln_dist;
        mtx_res.col_cnt = col_r + col_l + col_cnt + (col_cnt - 1) * col_dist;
        auto elem_cnt = mtx_res.ln_cnt * mtx_res.col_cnt;
        mtx_res.mtx_val = mtx_init(elem_cnt);
        for(auto i=0; i<ln_cnt; ++i)
            for(auto j=0; j<col_cnt; ++j)
            {
                auto curr_pad_no =  mtx_elem_pos(ln_t+i*(ln_dist+1), col_l+j*(col_dist+1), mtx_res.ln_cnt, mtx_res.col_cnt),
                    curr_origin_no = mtx_elem_pos(i, j, ln_cnt, col_cnt);
                mtx_res.mtx_val[curr_pad_no] = mtx_val[curr_origin_no];
            }
    }
    return mtx_res;
}

mtx_info mtx_crop(MATRIX &mtx_val, uint64_t ln_cnt, uint64_t col_cnt, uint64_t ln_t, uint64_t col_r, uint64_t ln_b, uint64_t col_l, uint64_t ln_dist = 0, uint64_t col_dist = 0)
{
    mtx_info mtx_res;
    if(mtx_val && ln_cnt && col_cnt)
    {
        mtx_res.ln_cnt = (ln_cnt - (ln_t + ln_b) + ln_dist) / (ln_dist + 1);
        mtx_res.col_cnt = (col_cnt - (col_r + col_l) + col_dist) / (col_dist + 1);
        auto elem_cnt = mtx_res.ln_cnt * mtx_res.col_cnt;
        mtx_res.mtx_val = mtx_init(elem_cnt);
        for(auto i=0; i<mtx_res.ln_cnt; ++i)
            for(auto j=0; j<mtx_res.col_cnt; ++j)
            {
                auto curr_res_no = mtx_elem_pos(i, j, mtx_res.ln_cnt, mtx_res.col_cnt),
                    curr_origin_no = mtx_elem_pos(ln_t+i*(ln_dist+1), col_l+j*(col_dist+1), ln_cnt, col_cnt);
                mtx_res.mtx_val[curr_res_no] = mtx_val[curr_origin_no];
            }
    }
    return mtx_res;
}

class matrix
{
protected:
    mtx_info info;
    uint64_t elem_cnt = 0;
    void _init(uint64_t ln_cnt, uint64_t col_cnt, bool init_ptr = false)
    {
        elem_cnt = ln_cnt * col_cnt;
        info.ln_cnt = ln_cnt;
        info.col_cnt = col_cnt;
        if(init_ptr) info.mtx_val = mtx_init(elem_cnt);
    }
    void _para_reset()
    {
        elem_cnt = 0;
        info.col_cnt = 0;
        info.ln_cnt = 0;
    }
public:
    static matrix blank_matrix() {return matrix();}
    bool is_matrix() {return info.col_cnt && info.ln_cnt && info.mtx_val && elem_cnt && elem_cnt==info.ln_cnt*info.col_cnt;}
    MATRIX ptr() {return mtx_copy(info.mtx_val, info.ln_cnt, info.col_cnt);}
    matrix() {}
    matrix(uint64_t ln_cnt, uint64_t col_cnt, double rand_upper = 0, double rand_lower = 1, double rand_acc = 1e-5)
    {
        if(ln_cnt && col_cnt)
        {
            _init(ln_cnt, col_cnt);
            if(rand_upper < rand_lower) info.mtx_val = mtx_init(elem_cnt);
            else info.mtx_val = mtx_init_rand(elem_cnt, rand_upper, rand_lower, rand_acc);
        }
    }
    matrix(MATRIX &&ptr_val, uint64_t ln_cnt, uint64_t col_cnt)
    {
        _init(ln_cnt, col_cnt);
        info.mtx_val = std::move(ptr_val);
    }
    matrix(MATRIX &ptr_val, uint64_t ln_cnt, uint64_t col_cnt)
    {
        _init(ln_cnt, col_cnt);
        info.mtx_val = mtx_copy(ptr_val, elem_cnt);
    }
    matrix(double atom) 
    {
        _init(1, 1, true);
        info.mtx_val[0] = atom;
    }
    matrix(matrix &val) {*this = val;}
    matrix(matrix &&val) {*this = std::move(val);}
    matrix(std::initializer_list<std::initializer_list<double>> _vect) {*this = _vect;}
    bool value_copy(matrix &val)
    {
        if(val.is_matrix())
        {
            _init(val.info.ln_cnt, val.info.col_cnt, elem_cnt!=val.elem_cnt);
            for(auto i=0; i<elem_cnt; ++i) info.mtx_val[i] = val.info.mtx_val[i];
            return true;
        }
        else return false;
    }
    bool value_move(matrix &&val)
    {
        if(val.is_matrix())
        {
            _init(val.info.ln_cnt, val.info.col_cnt);
            info.mtx_val = std::move(val.info.mtx_val);
            val._para_reset();
            return true;
        }
        else return false;
    }
    void value_fill(double val) {for(auto i=0; i<elem_cnt; ++i) info.mtx_val[i] = val;}
    uint64_t get_ln_cnt() {return info.ln_cnt;}
    uint64_t get_col_cnt() {return info.col_cnt;}
    uint64_t get_elem_cnt() {return elem_cnt;}
    __declspec(property(get = get_ln_cnt)) uint64_t LN_CNT;
    __declspec(property(get = get_col_cnt)) uint64_t COL_CNT;
    __declspec(property(get = get_elem_cnt)) uint64_t ELEM_CNT;
    double determinant()
    {
        if(info.ln_cnt == info.col_cnt) return mtx_det(info.mtx_val, info.ln_cnt);
        else return NAN;
    }
    matrix inverser()
    {
        if(info.ln_cnt == info.col_cnt) return matrix(mtx_inverser(info.mtx_val, info.ln_cnt), info.ln_cnt, info.col_cnt);
        else return matrix();
    }
    matrix transposition()
    {
        if(is_matrix()) return matrix(mtx_transposition(info.mtx_val, info.ln_cnt, info.col_cnt), info.col_cnt, info.ln_cnt);
        else return matrix();
    }
    mtx_extm extremum(uint64_t from_ln, uint64_t to_ln, uint64_t from_col, uint64_t to_col, uint64_t ln_dilation = 0, uint64_t col_dilation = 0, bool max_flag = true) {return mtx_extm_val(info.mtx_val, from_ln, to_ln, from_col, to_col, info.ln_cnt, info.col_cnt, ln_dilation, col_dilation, max_flag);}
    double atom()
    {
        if(elem_cnt==1 && is_matrix()) return info.mtx_val[0];
        else return NAN;
    }
    matrix child(uint64_t from_ln, uint64_t to_ln, uint64_t from_col, uint64_t to_col, uint64_t ln_dilation = 0, uint64_t col_dilation = 0) 
    {
        auto child_info = mtx_child_vec(info.mtx_val, from_ln, to_ln, from_col, to_col, ln_dilation, col_dilation);
        return matrix(std::move(child_info.mtx_val), child_info.ln_cnt, child_info.col_cnt);
    }
    matrix counter_clockwise_rotate_rect()
    {
        if(is_matrix()) return matrix(mtx_rotate_rect(info.mtx_val, info.ln_cnt, info.col_cnt), info.col_cnt, info.ln_cnt);
        else return matrix();
    }
    matrix mirror_flip()
    {
        if(is_matrix()) return matrix(mtx_mirror_flip(info.mtx_val, info.ln_cnt, info.col_cnt), info.col_cnt, info.ln_cnt);
        else return matrix();
    }
    bool shape_valid(uint64_t ln_cnt, uint64_t col_cnt)
    {
        if(ln_cnt==info.ln_cnt && col_cnt==info.col_cnt) return true;
        else return false;
    }
    bool shape_valid(matrix &mtx_src) {return shape_valid(mtx_src.info.ln_cnt, mtx_src.info.col_cnt);}
    matrix reshape(uint64_t ln_cnt, uint64_t col_cnt)
    {
        auto elem_cnt = ln_cnt * col_cnt;
        if(this->elem_cnt==elem_cnt && is_matrix()) return matrix(info.mtx_val, ln_cnt, col_cnt);
        else return matrix();
    }
    matrix reshape(matrix &as_val) {return reshape(as_val.info.ln_cnt, as_val.info.col_cnt);}
    double elem_sum(uint64_t from_ln, uint64_t to_ln, uint64_t from_col, uint64_t to_col, uint64_t ln_dilation = 0, uint64_t col_dilation = 0) {return mtx_sum(info.mtx_val, from_ln, to_ln, from_col, to_col, ln_dilation, col_dilation);}
    double elem_sum() {return elem_sum(0, info.ln_cnt-1, 0, info.col_cnt-1);}
    matrix abs()
    {
        auto cpy_ptr = mtx_copy(info.mtx_val, elem_cnt);
        mtx_abs(cpy_ptr, elem_cnt);
        return matrix(std::move(cpy_ptr), info.ln_cnt, info.col_cnt);
    }
    matrix elem_cal_opt(matrix &r_val, uint64_t opt_idx)
    {
        if(shape_valid(r_val)) return matrix(mtx_elem_cal_opt(info.mtx_val, r_val.info.mtx_val, info.ln_cnt, info.col_cnt, opt_idx), info.ln_cnt, info.col_cnt);
        else return matrix();
    }
    matrix elem_cal_opt(double para, uint64_t opt_idx)
    {
        if(is_matrix()) return matrix(mtx_elem_cal_opt(info.mtx_val, para, info.ln_cnt, info.col_cnt, opt_idx), info.ln_cnt, info.col_cnt);
        else return matrix();
    }
    matrix broadcast_add(double val) {return matrix(mtx_broadcast_add(info.mtx_val, elem_cnt, val), info.ln_cnt, info.col_cnt);}
    matrix broadcast_subtract(double val, bool is_subtrahend = true) {return matrix(mtx_broadcast_subtract(info.mtx_val, elem_cnt, val, is_subtrahend), info.ln_cnt, info.col_cnt);}
    matrix pad(uint64_t ln_t, uint64_t col_r, uint64_t ln_b, uint64_t col_l, uint64_t ln_dist = 0, uint64_t col_dist = 0)
    {
        auto pad_info = mtx_pad(info.mtx_val, info.ln_cnt, info.col_cnt, ln_t, col_r, ln_b, col_l, ln_dist, col_dist);
        return matrix(pad_info.mtx_val, pad_info.ln_cnt, pad_info.col_cnt);
    }
    matrix crop(uint64_t ln_t, uint64_t col_r, uint64_t ln_b, uint64_t col_l, uint64_t ln_dist = 0, uint64_t col_dist = 0)
    {
        auto crop_info = mtx_crop(info.mtx_val, info.ln_cnt, info.col_cnt, ln_t, col_r, ln_b, col_l, ln_dist, col_dist);
        return matrix(crop_info.mtx_val, crop_info.ln_cnt, crop_info.col_cnt);
    }
    matrix round_fit() 
    {
        if(is_matrix())
        {
            auto cpy_ptr = mtx_copy(info.mtx_val, elem_cnt);
            for(auto i=0; i<elem_cnt; ++i) cpy_ptr[i] = std::round(cpy_ptr[i]);
            return matrix(std::move(cpy_ptr), info.ln_cnt, info.col_cnt);
        }
        else return matrix();
    }
    matrix LU()
    {
        if(info.ln_cnt == info.col_cnt && is_matrix()) return matrix(mtx_LU(info.mtx_val, info.ln_cnt), info.ln_cnt, info.col_cnt);
        else return matrix();
    }
    matrix linear_eq(matrix &val_b, bool eq_idx = EQ_LU)
    {
        if(is_matrix() && info.ln_cnt == info.col_cnt && info.ln_cnt==val_b.info.ln_cnt && val_b.info.col_cnt==1)
            if(eq_idx) return matrix(mtx_equation(info.mtx_val, val_b.info.mtx_val, info.ln_cnt), info.ln_cnt, info.col_cnt);
            else return matrix(mtx_jacobi_iterate(info.mtx_val, val_b.info.mtx_val, info.ln_cnt), info.ln_cnt, info.col_cnt);
        else return matrix();
    }
    matrix operator+(matrix &val) {return matrix(mtx_add(info.mtx_val, val.info.mtx_val, info.ln_cnt, info.col_cnt), info.ln_cnt, info.col_cnt);}
    matrix operator-(matrix &val) {return matrix(mtx_add(info.mtx_val, val.info.mtx_val, info.ln_cnt, info.col_cnt, true), info.ln_cnt, info.col_cnt);}
    void operator+=(matrix &val) {new (this)matrix(std::move(*this + val));}
    void operator-=(matrix &val) {new (this)matrix(std::move(*this - val));}
    matrix operator*(matrix &val) {return matrix(mtx_mult(info.mtx_val, val.info.mtx_val, info.ln_cnt, info.col_cnt, val.info.ln_cnt, val.info.col_cnt), info.ln_cnt, val.info.col_cnt);}
    void operator*=(matrix &val) {new (this)matrix(std::move(*this * val));}
    matrix operator*(double val) {return matrix(mtx_mult(info.mtx_val, val, info.ln_cnt, info.col_cnt), info.ln_cnt, info.col_cnt);}
    void operator*=(double val) {new (this)matrix(std::move(*this * val));}
    friend matrix operator*(double val, matrix &r_val) {return r_val * val;}
    void operator=(matrix &val) {value_copy(val);}
    void operator=(matrix &&val) {value_move(std::move(val));}
    void operator=(std::initializer_list<std::initializer_list<double>> _vect)
    {
        bagrt::net_list<double> elem_temp;
        auto _col_cnt = 0;
        bool asg_flag = true;
        for(auto ln_temp : _vect)
        {
            auto col_cnt_temp = 0;
            for(auto col_temp : ln_temp)
            {
                ++ col_cnt_temp;
                elem_temp.emplace_back(col_temp);
            }
            if(!_col_cnt) _col_cnt = col_cnt_temp;
            else if(_col_cnt != col_cnt_temp)
            {
                asg_flag = false;
                break;
            }
            else continue;
        }
        if(asg_flag)
        {
            auto _elem_cnt = elem_temp.size();
            if(_elem_cnt != elem_cnt || _col_cnt != info.col_cnt)  _init(_elem_cnt/_col_cnt, _col_cnt, true);
            auto head = elem_temp.head_node();
            for(auto i=0; i<elem_cnt; ++i)
            {
                info.mtx_val[i] = head->data;
                head = head->next_node.get();
            }
        }
    }
    friend std::ostream &operator<<(std::ostream &output, matrix &out_matrix)
    {
        if(out_matrix.is_matrix()) for(auto i=0; i<out_matrix.elem_cnt; ++i)
        {
            output << out_matrix.info.mtx_val[i];
            if(!((i+1)%out_matrix.info.col_cnt || (i+1)==out_matrix.elem_cnt)) output << std::endl;
            else output << '\t'; 
        }
        return output;
    }
    ~matrix() {_para_reset();}
};

MATRIX_END