ADA_BEGIN

struct AdaDeltaVect
{
private:
    vect vecExpGrad;
    vect vecExpDelta;
public:
    double dRho = 0.95;
    double dEpsilon = 1e-6;
    vect vecPreDelta;
    AdaDeltaVect(){}
    AdaDeltaVect(AdaDeltaVect &advSrc) {*this = advSrc;}
    AdaDeltaVect(AdaDeltaVect &&advSrc) {*this = std::move(advSrc);}
    AdaDeltaVect(uint64_t nSizeLnCnt, uint64_t nSizeColCnt, double dRho = 0.95, double dEpsilon = 1e-6)
    {
        vecExpGrad = vect(nSizeLnCnt, nSizeColCnt);
        vecExpDelta = vect(nSizeLnCnt, nSizeColCnt);
        this->dRho = dRho;
        this->dEpsilon = dEpsilon;
    }
    void operator=(AdaDeltaVect &advSrc)
    {
        dRho = advSrc.dRho;
        dEpsilon = advSrc.dEpsilon;
        vecExpDelta = advSrc.vecExpDelta;
        vecExpGrad = advSrc.vecExpGrad;
    }
    void operator=(AdaDeltaVect &&advSrc)
    {
        dRho = advSrc.dRho;
        dEpsilon = advSrc.dEpsilon;
        vecExpDelta = std::move(advSrc.vecExpDelta);
        vecExpGrad = std::move(advSrc.vecExpGrad);
    }
    vect Delta(vect &vecCurrGrad)
    {
        if(!vecExpGrad.is_matrix()) vecExpGrad = vect(vecCurrGrad.get_ln_cnt(), vecCurrGrad.get_col_cnt());
        if(!vecExpDelta.is_matrix()) vecExpDelta = vecExpGrad;
        vecExpGrad = dRho * vecExpGrad + (1 - dRho) * vecCurrGrad.elem_cal_opt(2, MATRIX_ELEM_POW);
        auto vecRMSPreDelta = DIV_DOM(vecExpDelta, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW),
            vecRMSGrad = DIV_DOM(vecExpGrad, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW);
        vecPreDelta = vecRMSPreDelta.elem_cal_opt(vecRMSGrad, MATRIX_ELEM_DIV).elem_cal_opt(vecCurrGrad, MATRIX_ELEM_MULT);
        vecExpDelta = dRho * vecExpDelta + (1 - dRho) * vecPreDelta.elem_cal_opt(2, MATRIX_ELEM_POW);
        return vecPreDelta;
    }
    void Reset()
    {
        vecExpDelta.reset();
        vecExpGrad.reset();
        vecPreDelta.reset();
    }
    ~AdaDeltaVect() { Reset(); }
};

struct AdaDeltaVal
{
private:
    double dExpGrad = 0;
    double dExpDelta = 0;
public:
    double dRho = 0.95;
    double dEpsilon = 1e-6;
    double dPreDelta = 0;
    AdaDeltaVal(double dRhoVal = 0.95, double dEpsilonVal = 1e-6) : dRho(dRhoVal), dEpsilon(dEpsilonVal) {}
    AdaDeltaVal(AdaDeltaVal &advSrc) { *this = advSrc; }
    void operator=(AdaDeltaVal &advSrc)
    {
        dExpGrad = advSrc.dExpGrad;
        dExpDelta = advSrc.dExpDelta;
        dRho = advSrc.dRho;
        dEpsilon = advSrc.dEpsilon;
    }
    double Delta(double dCurrGrad)
    {
        dExpGrad = dRho * dExpGrad + (1 - dRho) * std::pow(dCurrGrad, 2);
        if(!dExpDelta) dExpDelta += dEpsilon;
        if(!dExpGrad) dExpGrad += dEpsilon;
        auto dRMSPreDelta = std::pow(dExpDelta, 0.5), dRMSCurrGrad = std::pow(dExpGrad, 0.5);
        dPreDelta = (dRMSPreDelta / dRMSCurrGrad) * dCurrGrad;
        dExpDelta = dRho * dExpDelta + (1 - dRho) * std::pow(dPreDelta, 2);
        return dPreDelta;
    }
    void Reset()
    {
        dExpGrad = 0;
        dExpDelta = 0;
    }
};

struct AdaNesterovVect
{
private:
    uint64_t iIterCnt = 0;
    double dEpsilon = 1e-6, dRho = 0.95, dNu = 0.95;
    vect vecM, vecN;
    void ValueAssign(AdaNesterovVect &anvSrc)
    {
        iIterCnt = anvSrc.iIterCnt;
        dEpsilon = anvSrc.dEpsilon;
        dRho = anvSrc.dRho;
        dNu = anvSrc.dNu;
    }
public:
    AdaNesterovVect() {}
    AdaNesterovVect(AdaNesterovVect &anvSrc) { *this = anvSrc; }
    AdaNesterovVect(AdaNesterovVect &&anvSrc) { *this = std::move(anvSrc); }
    void operator=(AdaNesterovVect &anvSrc)
    {
        ValueAssign(anvSrc);
        vecM = anvSrc.vecM;
        vecN = anvSrc.vecN;
    }
    void operator=(AdaNesterovVect &&anvSrc)
    {
        ValueAssign(anvSrc);
        vecM = std::move(anvSrc.vecM);
        vecN = std::move(anvSrc.vecN);
    }

    vect Momentum(vect &vecGrad, double dLearnRate)
    {
        if(!vecM.is_matrix())
        {
            vecM = vect(vecGrad.LN_CNT, vecGrad.COL_CNT);
            vecM.value_fill(dEpsilon);
        }
        if(!vecN.is_matrix()) vecN = vecM;
        vecM = dRho * vecM + (1 - dRho) * vecGrad;
        vecN = dNu * vecN + (1 - dNu) * vecGrad.elem_cal_opt(2, MATRIX_ELEM_POW);
        auto vecBarM = (1 / (1 - std::pow(dRho, ++iIterCnt))) * vecM,
            vecBarN = (1 / (1 - std::pow(dNu, iIterCnt))) * vecN;
        return dLearnRate * (dRho * vecBarM + ((1 - dRho) / (1 - std::pow(dRho, iIterCnt))) * vecGrad).elem_cal_opt(vecBarN.elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    }

    void Reset()
    {
        iIterCnt = 0;
        dEpsilon = 1e-6;
        dRho = 0.95;
        dNu = 0.95;
        vecM.reset();
        vecN.reset();
    }
    ~AdaNesterovVect() { Reset(); }
};

struct AdaNesterovVal
{
private:
    uint64_t iIterCnt = 0;
    double dEpsilon = 1e-6, dRho = 0.95, dNu = 0.95, dM = dEpsilon, dN = dEpsilon;
public:
    AdaNesterovVal() {}
    AdaNesterovVal(AdaNesterovVal &anvSrc) { *this = anvSrc; }
    void operator=(AdaNesterovVal &anvSrc)
    {
        iIterCnt = anvSrc.iIterCnt;
        dEpsilon = anvSrc.dEpsilon;
        dRho = anvSrc.dRho; dNu = anvSrc.dNu;
        dM = anvSrc.dM; dN = anvSrc.dN;
    }

    double Momentum(double dGrad, double dLearnRate)
    {
        dM = dRho * dM + (1 - dRho) * dGrad;
        dN = dNu * dN + (1 - dNu) * std::pow(dGrad, 2);
        auto dBarM = (1 / (1 - std::pow(dRho, ++iIterCnt))) * dM,
            dBarN = (1 / (1 - std::pow(dNu, iIterCnt))) * dN;
        return (dLearnRate / std::pow(dBarN, 0.5)) * (dRho * dBarM + ((1 - dRho) * dGrad) / (1 - std::pow(dRho, iIterCnt)));
    }

    void Reset()
    {
        iIterCnt = 0; dEpsilon = 1e-6;
        dRho = 0.95; dNu = 0.95;
        dM = dEpsilon; dN = dEpsilon;
    }
};

ADA_END

FC_BEGIN

vect AdaDeltaUpdateWeight(vect &vecWeight, vect &vecGradLossToWeight, ada::AdaDeltaVect &advCurrLayerDelta)
{
    auto vecCurrDelta = advCurrLayerDelta.Delta(vecGradLossToWeight);
    if(vecCurrDelta.is_matrix()) return vecWeight - vecCurrDelta;
    else return blank_vect;
}

double BNAdaDeltaUpdateScaleShift(double dGammaBeta, double dGradLossToScaleShift, ada::AdaDeltaVal &advCurrDelta) { return dGammaBeta - advCurrDelta.Delta(dGradLossToScaleShift); }

vect AdaNesterovUpdateWeight(vect &vecWeight, vect &vecGradLossToWeight, double dLearnRate, ada::AdaNesterovVect &anvCurrLayerMomt)
{
    auto vecCurrMomt = anvCurrLayerMomt.Momentum(vecGradLossToWeight, dLearnRate);
    if(vecCurrMomt.is_matrix()) return vecWeight - vecCurrMomt;
    else return blank_vect;
}

double BNAdaNesterovUpdateScaleShift(double dGammaBeta, double dGradLossToScaleShift, double dLearnRate, ada::AdaNesterovVal &anvCurrMomt) { return dGammaBeta - anvCurrMomt.Momentum(dGradLossToScaleShift, dLearnRate); }

FC_END

CONV_BEGIN

tensor AdaDeltaUpdateKernel(tensor &tenKernel, tensor &tenGradLossToKernel, ada::ada_tensor<ada::AdaDeltaVect> &advCurrDelta)
{
    if(tenKernel.size() == tenGradLossToKernel.size())
    {
        tensor tenUpdatedKernel(tenKernel.size());
        thrd_pool::thread_pool t_p;
        for(auto i=0; i<tenKernel.size(); ++i)
            if(tenKernel[i].size() == tenGradLossToKernel[i].size())
            {
                tenUpdatedKernel[i] = t_p.add_task([&]
                {
                    feature vecTemp(tenKernel[i].size());
                    for(auto j=0; j<tenKernel[i].size(); ++j)
                    {
                        vecTemp[j] = tenKernel[i][j] - advCurrDelta[i][j].Delta(tenGradLossToKernel[i][j]);
                        if(!vecTemp[j].is_matrix()) return blank_feature;
                    }
                    return vecTemp;
                }).get();
            }
            else return blank_tensor;
        return tenUpdatedKernel;
    }
    else return blank_tensor;
}

vect BNAdaDeltaUpdateScaleShift(vect &vecGammaBeta, vect &vecGradLossToScaleShift, ada::AdaDeltaVect &advCurrDelta)
{
    if(vecGammaBeta.shape_valid(vecGradLossToScaleShift)) return vecGammaBeta - advCurrDelta.Delta(vecGradLossToScaleShift);
    else return blank_vect;
}

tensor AdaNesterovUpdateKernel(tensor &tenKernel, tensor &tenGradLossToKernel, double dLearnRate, ada::ada_tensor<ada::AdaNesterovVect> &anvCurrMomt)
{
    if(tenKernel.size() == tenGradLossToKernel.size())
    {
        tensor tenUpdatedKernel(tenKernel.size());
        for(auto i=0; i<tenKernel.size(); ++i)
            if(tenKernel[i].size() == tenGradLossToKernel[i].size())
            {
                tenUpdatedKernel[i].init(tenKernel[i].size());
                for(auto j=0; j<tenKernel[i].size(); ++j)
                {
                    tenUpdatedKernel[i][j] = tenKernel[i][j] - anvCurrMomt[i][j].Momentum(tenGradLossToKernel[i][j], dLearnRate);
                    if(!tenUpdatedKernel[i][j].is_matrix()) return blank_tensor;
                }
            }
            else return blank_tensor;
        return tenUpdatedKernel;
    }
    else return blank_tensor;
}

vect BNAdaNesterovUpdateScaleShift(vect &vecGammaBeta, vect &vecGradLossToScaleShift, double dLearnRate, ada::AdaNesterovVect &anvCurrMomt)
{
    if(vecGammaBeta.shape_valid(vecGradLossToScaleShift)) return vecGammaBeta - anvCurrMomt.Momentum(vecGradLossToScaleShift, dLearnRate);
    else return blank_vect;
}

CONV_END