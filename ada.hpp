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
    friend std::ostream& operator<<(std::ostream &output, AdaDeltaVect &val)
    {
        output << "E[Gradient] = " << val.vecExpGrad << "; E[Delta] = " << val.vecExpDelta << std::endl;
        output<< "Rho = " << val.dRho << "; Delta = " << val.vecPreDelta << std::endl;
        return output;
    }
    AdaDeltaVect(){}
    AdaDeltaVect(AdaDeltaVect &asSrc) {*this = asSrc;}
    AdaDeltaVect(AdaDeltaVect &&asSrc) {*this = std::move(asSrc);}
    AdaDeltaVect(uint64_t nSizeLnCnt, uint64_t nSizeColCnt, double dRho = 0.95, double dEpsilon = 1e-6)
    {
        vecExpGrad = vect(nSizeLnCnt, nSizeColCnt);
        vecExpGrad = vect(nSizeLnCnt, nSizeColCnt);
        this->dRho = dRho;
        this->dEpsilon = dEpsilon;
    }
    void operator=(AdaDeltaVect &adSrc)
    {
        dRho = adSrc.dRho;
        dEpsilon = adSrc.dEpsilon;
        vecExpDelta = adSrc.vecExpDelta;
        vecExpGrad = adSrc.vecExpGrad;
    }
    void operator=(AdaDeltaVect &&adSrc)
    {
        dRho = adSrc.dRho;
        dEpsilon = adSrc.dEpsilon;
        vecExpDelta = std::move(adSrc.vecExpDelta);
        vecExpGrad = std::move(adSrc.vecExpGrad);
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
    // ~AdaDeltaVect() {}
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
    friend std::ostream& operator<<(std::ostream &output, AdaDeltaVal &val)
    {
        output << "E[Gradient] = " << val.dExpGrad << "; E[Delta] = " << val.dExpDelta << std::endl;
        output<< "Rho = " << val.dRho << "; Delta = " << val.dPreDelta << std::endl;
        return output;
    }
    AdaDeltaVal(double dRhoVal = 0.95, double dEpsilonVal = 1e-6) : dRho(dRhoVal), dEpsilon(dEpsilonVal) {}
    void operator=(AdaDeltaVal &adSrc)
    {
        dExpGrad = adSrc.dExpGrad;
        dExpDelta = adSrc.dExpDelta;
        dRho = adSrc.dRho;
        dEpsilon = adSrc.dEpsilon;
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
    // ~AdaDeltaVal()
    // {
    //     dExpGrad = 0;
    //     dExpDelta = 0;
    // }
};

ADA_END

FC_BEGIN

vect AdaDeltaUpdateWeight(vect &vecWeight, vect &vecGradLossToWeight, ada::AdaDeltaVect &advCurrLayerDelta)
{
    auto vecCurrDelta = advCurrLayerDelta.Delta(vecGradLossToWeight);
    if(vecCurrDelta.is_matrix()) return vecWeight - vecCurrDelta;
    else return blank_vect;
}

double BNAdaDeltaUpdateScaleShift(double dGammaBeta, double dGradLossToScaleShift, ada::AdaDeltaVal advCurrDelta) {return dGammaBeta - advCurrDelta.Delta(dGradLossToScaleShift);}

FC_END

CONV_BEGIN

tensor AdaDeltaUpdateKernel(tensor &tenKernel, tensor &tenGradLossToKernel, ada::ada_tensor<ada::AdaDeltaVect> &advCurrDelta)
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
                    tenUpdatedKernel[i][j] = tenKernel[i][j] - advCurrDelta[i][j].Delta(tenGradLossToKernel[i][j]);
                    if(!tenUpdatedKernel[i][j].is_matrix()) return blank_tensor;
                }
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

CONV_END