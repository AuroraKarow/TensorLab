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
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
    {
        setGradLossToInput[i] = GradLossToInput(setGradLossToOutput[i], vecWeight);
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

struct FCBN : BN
{
    vect vecMiuBeta;
    vect vecSigmaSqr;
    set<vect> setBarX;
    set<vect> setY;
    FCBN(){}
    FCBN(FCBN &FCBNVal) 
    {
        vecMiuBeta = FCBNVal.vecMiuBeta;
        vecSigmaSqr = FCBNVal.vecSigmaSqr;
        setBarX = FCBNVal.setBarX;
        setY = FCBNVal.setY;
    }
    FCBN(FCBN &&FCBNVal)
    {
        vecMiuBeta = std::move(FCBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(FCBNVal.vecSigmaSqr);
        setBarX = std::move(FCBNVal.setBarX);
        setY = std::move(FCBNVal.setY);
    }
    void operator=(FCBN &FCBNVal) {new(this)FCBN(FCBNVal);}
    void operator=(FCBN &&FCBNVal) {new(this)FCBN(std::move(FCBNVal));}
    // ~FCBN() {}
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

double BNUpdateScaleShift(double dScaleShift, double dGradLossToScaleShift, double dLearnRate) {return dScaleShift - dLearnRate * dGradLossToScaleShift;}

bagrt::net_queue<vect> BNDeduce(bagrt::net_queue<vect> &setNetInput, double dBeta, double dGamma, set<BN_PTR> &setbnData, uint64_t iMiniBatchSize = 0, double dEpsilon = 1e-10)
{
    /**
     * Expectation Average, Expectation MiuBeta
     * Variance mini-batch variance, Variance SigmaSqr
     */
    vect vecEX = INSTANCE_DERIVE<FCBN>(setbnData[0]) -> vecMiuBeta,
        vecEVarX = INSTANCE_DERIVE<FCBN>(setbnData[0]) -> vecSigmaSqr;
    for(auto i=1; i<setbnData.size(); ++i)
    {
        vecEX += INSTANCE_DERIVE<FCBN>(setbnData[i]) -> vecMiuBeta;
        vecEVarX += INSTANCE_DERIVE<FCBN>(setbnData[i]) -> vecSigmaSqr;
    }
    if(setbnData.size() > 1)
    {
        vecEX = vecEX.elem_cal_opt(setbnData.size(), MATRIX_ELEM_DIV);
        vecEVarX = (iMiniBatchSize / (iMiniBatchSize - 1)) * vecEVarX.elem_cal_opt(setbnData.size(), MATRIX_ELEM_DIV);
    }
    // Normalize
    set<vect> setBNDeduceOutput(setNetInput.size());
    for(auto i=0; i<setNetInput.size(); ++i)
    {
        auto vecBarX = (setNetInput[i] - vecEX).elem_cal_opt(DIV_DOM(vecEVarX, dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
        setBNDeduceOutput[i] = (dGamma * vecBarX).broadcast_add(dBeta);
    }
    return setBNDeduceOutput;
}

FC_END