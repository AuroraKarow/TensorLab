ADA_BEGIN

struct AdaDeltaVect
{
private:
    vect vecExpGrad;
    vect vecExpDelta;
public:
    double dRho = 0.95;
    double dEpsilon = 1e-3;
    vect vecPreDelta;
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
    double dEpsilon = 1e-3;
    double dPreDelta = 0;
    AdaDeltaVal(double dExpGradVal = 0, double dExpDeltaVal = 0, double dRhoVal = 0.95, double dEpsilonVal = 1e-6) :
    dExpGrad(dExpGradVal), dExpDelta(dExpDeltaVal), dRho(dRhoVal), dEpsilon(dEpsilonVal) {}
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

double AdaDeltaUpdateScaleShift(double dGammaBeta, double dGradLossToScaleShift, ada::AdaDeltaVal advCurrDelta) {return dGammaBeta - advCurrDelta.Delta(dGradLossToScaleShift);}

FC_END

CONV_BEGIN



CONV_END