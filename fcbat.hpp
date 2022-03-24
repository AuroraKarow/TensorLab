FC_BEGIN

vect InitWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    if(iInputLnCnt && iOutputLnCnt) return vect(iOutputLnCnt, iInputLnCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    else return blank_vect;
}

set<vect> Output(set<vect> &setInput, vect &vecWeight)
{
    set<vect> setOutput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOutput[i] = Output(setInput[i], vecWeight);
        if(!setOutput[i].is_matrix()) return blank_vect_seq;
    }
    return setOutput;
}

set<vect> GradLossToInput(set<vect> &setGradLossToOutput, vect &vecWeight)
{
    set<vect> setGradLossToInput(setGradLossToOutput.size());
    thrd_pool::thread_pool t_p;
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
    {
        setGradLossToInput[i] = t_p.add_task([&]{ return GradLossToInput(setGradLossToOutput[i], vecWeight); }).get();
        if(!setGradLossToInput[i].is_matrix()) return blank_vect_seq;
    }
    return setGradLossToInput;
}

vect GradLossToWeight(set<vect> &setGradLossToOutput, set<vect> &setInput)
{
    vect vecGradLossToWeight;
    if(setGradLossToOutput.size()==setInput.size()) for(auto i=0; i<setInput.size(); ++i)
    {
        if(vecGradLossToWeight.is_matrix()) vecGradLossToWeight += GradLossToWeight(setGradLossToOutput[i], setInput[i]);
        else vecGradLossToWeight = GradLossToWeight(setGradLossToOutput[i], setInput[i]);
        if(!vecGradLossToWeight.is_matrix()) break;
    }
    return vecGradLossToWeight;
}

set<vect> FeatureTransform(set<feature> &setInput)
{
    set<vect> setOuput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOuput[i] = FeatureTransform(setInput[i]);
        if(!setOuput[i].is_matrix()) return blank_vect_seq;
    }
    return setOuput;
}

set<feature> FeatureTransform(set<vect> &setInput, uint64_t iLnCnt, uint64_t iColCnt)
{
    set<feature> setOuput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOuput[i] = FeatureTransform(setInput[i], iLnCnt, iColCnt);
        if(!setOuput[i].size()) return blank_ft_seq;
    }
    return setOuput;
}

set<vect> FeatureTransformIm2Col(set<vect> &setInput)
{
    set<vect> setOuput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOuput[i] = FeatureTransformIm2Col(setInput[i]);
        if(!setOuput[i].is_matrix()) return blank_vect_seq;
    }
    return setOuput;
}

set<vect> FeatureTransformIm2Col(set<vect> &setInput, uint64_t iLnCnt, uint64_t iColCnt)
{
    set<vect> setOuput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOuput[i] = FeatureTransformIm2Col(setInput[i], iLnCnt, iColCnt);
        if(!setOuput[i].is_matrix()) return blank_vect_seq;
    }
    return setOuput;
}

struct FCBN : BN
{
    vect vecMiuBeta;
    vect vecSigmaSqr;
    set<vect> setBarX;
    set<vect> setY;
    FCBN(){}
    void ValueCopy(FCBN &FCBNVal)
    {
        vecMiuBeta = FCBNVal.vecMiuBeta;
        vecSigmaSqr = FCBNVal.vecSigmaSqr;
        setBarX = FCBNVal.setBarX;
        setY = FCBNVal.setY;
    }
    void ValueMove(FCBN &&FCBNVal)
    {
        vecMiuBeta = std::move(FCBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(FCBNVal.vecSigmaSqr);
        setBarX = std::move(FCBNVal.setBarX);
        setY = std::move(FCBNVal.setY);
    }
    FCBN(FCBN &FCBNVal) { ValueCopy(FCBNVal); }
    FCBN(FCBN &&FCBNVal) { ValueMove(std::move(FCBNVal)); }
    void operator=(FCBN &FCBNVal) { ValueCopy(FCBNVal); }
    void operator=(FCBN &&FCBNVal) { ValueMove(std::move(FCBNVal)); }
    void reset()
    {
        vecMiuBeta.reset();
        vecSigmaSqr.reset();
        setBarX.reset();
        setY.reset();
    }
    ~FCBN() { reset(); }
};

FCBN BNTrain(set<vect> &setInput, double dBeta = 0, double dGamma = 1, double dEpsilon = 1e-10)
{
    FCBN BNOutput;
    // Average, miu
    for(auto i=0; i<setInput.size(); ++i) if(BNOutput.vecMiuBeta.is_matrix()) BNOutput.vecMiuBeta += setInput[i];
    else BNOutput.vecMiuBeta = setInput[i];
    BNOutput.vecMiuBeta = BNOutput.vecMiuBeta.elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    // Variance, sigma square
    for(auto i=0; i<setInput.size(); ++i)
    {
        auto vecSglVarc = (setInput[i] - BNOutput.vecMiuBeta).elem_cal_opt(2, MATRIX_ELEM_POW);
        if(BNOutput.vecSigmaSqr.is_matrix()) BNOutput.vecSigmaSqr += vecSglVarc;
        else BNOutput.vecSigmaSqr = vecSglVarc;
    }
    BNOutput.vecSigmaSqr = BNOutput.vecSigmaSqr.elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    // Normalize, bar x
    BNOutput.setBarX.init(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
        BNOutput.setBarX[i] = (setInput[i] - BNOutput.vecMiuBeta).elem_cal_opt(DIV_DOM(BNOutput.vecSigmaSqr, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    // Scale shift, y
    BNOutput.setY.init(setInput.size());
    for(auto i=0; i<setInput.size(); ++i) BNOutput.setY[i] = (dGamma * BNOutput.setBarX[i]).broadcast_add(dBeta);
    return BNOutput;
}

set<vect> BNGradLossToInput(FCBN &FCBNOutput, set<vect> &setInput, set<vect> &setGradLossToOutput, double dGamma, double dEpsilon = 1e-10)
{
    // Operation value
    auto vecDmtSigmaSqr = DIV_DOM(FCBNOutput.vecSigmaSqr, dEpsilon);
    auto vecDmtSigma = vecDmtSigmaSqr.elem_cal_opt(0.5, MATRIX_ELEM_POW);
    // Gradient loss to normalized output, bar x
    set<vect> setGradLossToBarX(setInput.size());
    for(auto i=0; i<setInput.size(); ++i) setGradLossToBarX[i] = setGradLossToOutput[i] * dGamma;
    // Gradient loss to variance, square-powered sigma
    vect vecGradLossToSigmaSqr(FCBNOutput.vecSigmaSqr.get_ln_cnt(), FCBNOutput.vecSigmaSqr.get_col_cnt());
    for(auto i=0; i<setInput.size(); ++i) vecGradLossToSigmaSqr += (-1) * setGradLossToBarX[i].elem_cal_opt((setInput[i] - FCBNOutput.vecMiuBeta),  MATRIX_ELEM_MULT).elem_cal_opt((2 * vecDmtSigmaSqr.elem_cal_opt(1.5,  MATRIX_ELEM_POW)),  MATRIX_ELEM_DIV);
    // Gradient loss to average, miubeta
    vect vecGradLossToMiuBeta(FCBNOutput.vecMiuBeta.get_ln_cnt(), FCBNOutput.vecMiuBeta.get_col_cnt());
    for(auto i=0; i<setInput.size(); ++i) vecGradLossToMiuBeta += (-1) * setGradLossToBarX[i].elem_cal_opt(vecDmtSigma,  MATRIX_ELEM_DIV);
    vect vecAvgMidDist(FCBNOutput.vecMiuBeta.get_ln_cnt(), FCBNOutput.vecMiuBeta.get_col_cnt());
    for(auto i=0; i<setInput.size(); ++i) vecAvgMidDist += (-2) * (setInput[i] - FCBNOutput.vecMiuBeta);
    vecAvgMidDist = vecAvgMidDist.elem_cal_opt(setInput.size(),  MATRIX_ELEM_DIV);
    vecGradLossToMiuBeta += vecGradLossToSigmaSqr.elem_cal_opt(vecAvgMidDist,  MATRIX_ELEM_MULT);
    // Gradient loss to input, x
    set<vect> setGradLossToInput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
        setGradLossToInput[i] = setGradLossToBarX[i].elem_cal_opt(vecDmtSigma,  MATRIX_ELEM_DIV) + vecGradLossToSigmaSqr.elem_cal_opt((2.0 / setInput.size()) * (setInput[i] - FCBNOutput.vecMiuBeta),  MATRIX_ELEM_MULT) + vecGradLossToMiuBeta.elem_cal_opt(setInput.size(),  MATRIX_ELEM_DIV);
    return setGradLossToInput;
}

double BNGradLossToScale(set<vect> &setGradLossToOutput, FCBN &FCBNOutput)
{
    double dGrad = 0;
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
        for(auto j=0; j<setGradLossToOutput[i].ELEM_CNT; ++j) dGrad += setGradLossToOutput[i].pos_idx(j) * FCBNOutput.setBarX[i].pos_idx(j);
    return dGrad;
}

double BNGradLossToShift(set<vect> &setGradLossToOutput)
{
    double dGrad = 0;
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
        for(auto j=0; j<setGradLossToOutput[i].ELEM_CNT; ++j) dGrad += setGradLossToOutput[i].pos_idx(j);
    return dGrad;
}

vect BNDeduce(vect &vecInput, double dBeta, double dGamma, std::shared_ptr<FCBN> &pBNData, uint64_t iMiniBatchSize = 0, double iMiniBatchCnt = 0, double dEpsilon = 1e-10)
{
    /**
     * Expectation Average, Expectation MiuBeta
     * Variance mini-batch variance, Variance SigmaSqr
     */
    if(iMiniBatchCnt)
    {
        pBNData->vecMiuBeta = pBNData->vecMiuBeta.elem_cal_opt(iMiniBatchCnt, MATRIX_ELEM_DIV);
        pBNData->vecSigmaSqr = (iMiniBatchSize / (iMiniBatchSize - 1)) * pBNData->vecSigmaSqr.elem_cal_opt(iMiniBatchCnt, MATRIX_ELEM_DIV);
    }
    // Normalize
    auto vecBarX = (vecInput - pBNData->vecMiuBeta).elem_cal_opt(DIV_DOM(pBNData->vecSigmaSqr, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
    return (dGamma * vecBarX).broadcast_add(dBeta);
}

FC_END