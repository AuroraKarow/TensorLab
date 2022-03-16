CONV_BEGIN

tensor InitKernel(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    tensor tenKernel(iAmt);
    for(auto i=0; i<iAmt; ++i)
    {
        tenKernel[i].init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j) tenKernel[i][j] = vect(iLnCnt, iColCnt, true, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    }
    return tenKernel;
}

set<feature> Conv(set<feature> &setInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    set<feature> setOutput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOutput[i] = Conv(setInput[i], tenKernel, iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!setOutput[i].size()) return blank_ft_seq;
    }
    return setOutput;
}

tensor GradLossToKernel(set<feature> &setGradLossToOutput, set<feature> &setInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    tensor tenGradLossToKernel;
    for(auto i=0; i<setInput.size(); ++i)
    {
        auto tenSglGrad = GradLossToKernel(setGradLossToOutput[i], setInput[i], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(tenGradLossToKernel.size()) for(auto j=0; j<tenSglGrad.size(); ++j) for(auto k=0; k<tenSglGrad[i].size(); ++k) tenGradLossToKernel[j][k] += tenSglGrad[j][k];
        else tenGradLossToKernel = std::move(tenSglGrad);
        if(!tenGradLossToKernel.size()) return blank_tensor;
    }
    return tenGradLossToKernel;
}

tensor UpdateKernel(tensor &tenKernel, tensor tenGradLossToKernel, double dLearnRate)
{
    if(tenKernel.size() == tenGradLossToKernel.size())
    {
        tensor tenUpdatedKernel(tenKernel.size());
        for(auto i=0; i<tenKernel.size(); ++i) if(tenKernel[i].size() == tenGradLossToKernel.size())
        {
            tenUpdatedKernel[i].init(tenKernel[i].size());
            for(auto j=0; j<tenKernel[j].size(); ++j)
            {
                tenUpdatedKernel[i][j] = tenKernel[i][j] - dLearnRate * tenGradLossToKernel[i][j];
                if(!tenUpdatedKernel[i][j].is_matrix()) return blank_tensor;
            }
        }
        else return blank_tensor;
        return tenUpdatedKernel;
    }
    else return blank_tensor;
}

set<feature> GradLossToInput(set<feature> &setGradLossToOutput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    set<feature> setGradLossToInput(setGradLossToOutput.size());
    for(auto i=0; i<setGradLossToOutput.size(); ++i)
    {
        setGradLossToInput[i] = GradLossToInput(setGradLossToOutput[i], tenKernel, iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!setGradLossToInput[i].size()) return blank_ft_seq;
    }
    return setGradLossToInput;
}

set<feature> Pool(set<feature> &vecInput, uint64_t iPoolType = POOL_DOWN_MAX, bool bDownSamp = true, set<feature> &setTraceInput = set<feature>(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    set<feature> setOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i)
    {
        if(bDownSamp) setOutput[i] = PoolDown(vecInput[i], iPoolType, iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);
        else setOutput[i] = PoolUp(vecInput[i], iPoolType, setTraceInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);
        if(!setOutput[i].size()) return blank_ft_seq;
    }
    return setOutput;
}

struct ConvBN : BN
{
    feature vecMiuBeta;
    feature vecSigmaSqr;
    set<feature> setBarX;
    set<feature> setY;
    ConvBN(){}
    void ValueCopy(ConvBN &ConvBNVal)
    {
        vecMiuBeta = ConvBNVal.vecMiuBeta;
        vecSigmaSqr = ConvBNVal.vecSigmaSqr;
        setBarX = ConvBNVal.setBarX;
        setY = ConvBNVal.setY;
    }
    void ValueMove(ConvBN &&ConvBNVal)
    {
        vecMiuBeta = std::move(ConvBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(ConvBNVal.vecSigmaSqr);
        setBarX = std::move(ConvBNVal.setBarX);
        setY = std::move(ConvBNVal.setY);
    }
    ConvBN(ConvBN &ConvBNVal) { ValueCopy(ConvBNVal); }
    ConvBN(ConvBN &&ConvBNVal) { ValueMove(std::move(ConvBNVal)); }
    void operator=(ConvBN &ConvBNVal) { ValueCopy(ConvBNVal); }
    void operator=(ConvBN &&ConvBNVal) { ValueMove(std::move(ConvBNVal)); }
    void Reset()
    {
        vecMiuBeta.reset();
        vecSigmaSqr.reset();
        setBarX.reset();
        setY.reset();
    }
    ~ConvBN() { Reset(); }
};

vect BNInitScaleShift(uint64_t iChannCnt, double dFillVal)
{
    vect vecSS(iChannCnt, IDX_SGL);
    if(dFillVal) for(auto i=0; i<iChannCnt; ++i) vecSS.pos_idx(i) = dFillVal;
    return vecSS;
}

ConvBN BNTrain(set<feature> &setInput, vect &vecBeta, vect &vecGamma, double dEpsilon = 1e-5)
{
    ConvBN BNOutput;
    // Average & Variance
    BNOutput.vecMiuBeta.init(setInput[IDX_ZERO].size());
    BNOutput.vecSigmaSqr.init(setInput[IDX_ZERO].size());
    for(auto i=0; i<setInput[IDX_ZERO].size(); ++i)
    {
        for(auto j=0; j<setInput.size(); ++j)
            if(BNOutput.vecMiuBeta[i].is_matrix()) BNOutput.vecMiuBeta[i] += setInput[j][i];
            else BNOutput.vecMiuBeta[i] = setInput[j][i];
        BNOutput.vecMiuBeta[i] = BNOutput.vecMiuBeta[i].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
        for(auto j=0; j<setInput.size(); ++j)
        {
            auto vecSglSigmaSqr = (setInput[j][i] - BNOutput.vecMiuBeta[i]).elem_cal_opt(2, MATRIX_ELEM_POW);
            if(BNOutput.vecSigmaSqr[i].is_matrix()) BNOutput.vecSigmaSqr[i] += vecSglSigmaSqr;
            else BNOutput.vecSigmaSqr[i] = std::move(vecSglSigmaSqr);
        }
        BNOutput.vecSigmaSqr[i] = BNOutput.vecSigmaSqr[i].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    }
    // Normalize & Output
    BNOutput.setBarX.init(setInput.size());
    BNOutput.setY.init(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        BNOutput.setBarX[i].init(setInput[i].size());
        for(auto j=0; j<setInput[i].size(); ++j) BNOutput.setBarX[i][j] = (setInput[i][j] - BNOutput.vecMiuBeta[j]).elem_cal_opt(DIV_DOM(BNOutput.vecSigmaSqr[j], dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
        BNOutput.setY[i].init(setInput[i].size());
        for(auto j=0; j<setInput[i].size(); ++j) BNOutput.setY[i][j] = (vecGamma.pos_idx(j) * BNOutput.setBarX[i][j]).broadcast_add(vecBeta.pos_idx(j));
    }
    return BNOutput;
}

set<feature> BNGradLossToInput(ConvBN &ConvBNOutput, set<feature> &setInput, set<feature> &setGradLossToOutput, vect &vecGamma, double dEpsilon = 1e-5)
{
    // Sigma & Square-powered sigma
    feature vecDmrSigmaSqr(ConvBNOutput.vecSigmaSqr.size()), vecDmrSigma(ConvBNOutput.vecSigmaSqr.size());
    for(auto i=0; i<ConvBNOutput.vecSigmaSqr.size(); ++i)
    {
        vecDmrSigmaSqr[i] = DIV_DOM(ConvBNOutput.vecSigmaSqr[i], dEpsilon);
        vecDmrSigma[i] = vecDmrSigmaSqr[i].elem_cal_opt(0.5, MATRIX_ELEM_POW);
    }
    // Gradient loss to normalized output
    set<feature> setGradLossToBarX(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setGradLossToBarX[i].init(setInput[i].size());
        for(auto j=0; j<ConvBNOutput.vecMiuBeta.size(); ++j) setGradLossToBarX[i][j] = setGradLossToOutput[i][j] * vecGamma.pos_idx(j);
    }
    // Gradient loss to square-powered sigma
    feature vecGradLossToSigmaSqr(ConvBNOutput.vecSigmaSqr.size());
    for(auto i=0; i<ConvBNOutput.vecSigmaSqr.size(); ++i) for(auto j=0; j<setInput.size(); ++j)
    {
        auto vecSglGradLossToSigmaSqr = ((-1) * setGradLossToBarX[j][i].elem_cal_opt((setInput[j][i] - ConvBNOutput.vecMiuBeta[i]), MATRIX_ELEM_MULT)).elem_cal_opt((2 * vecDmrSigmaSqr[i].elem_cal_opt(1.5, MATRIX_ELEM_POW)), MATRIX_ELEM_DIV);
        if(vecGradLossToSigmaSqr[i].is_matrix()) vecGradLossToSigmaSqr[i] += vecSglGradLossToSigmaSqr;
        else vecGradLossToSigmaSqr[i] = std::move(vecSglGradLossToSigmaSqr);
    }
    // Gradient loss to miubeta
    feature vecGradLossToMiuBeta(ConvBNOutput.vecMiuBeta.size());
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
    {
        vect vecDistribute;
        vect vecDistance;
        for(auto j=0; j<setInput.size(); ++j)
        {
            auto vecSglDistribute = (-1) * setGradLossToBarX[j][i].elem_cal_opt(vecDmrSigma[i], MATRIX_ELEM_DIV);
            if(vecDistribute.is_matrix()) vecDistribute += vecSglDistribute;
            else vecDistribute = vecSglDistribute;
            auto vecSglDistance = (-2) * (setInput[j][i] - ConvBNOutput.vecMiuBeta[i]);
            if(vecDistance.is_matrix()) vecDistance += vecSglDistance;
            else vecDistance = vecSglDistance;
        }
        vecGradLossToMiuBeta[i] = vecDistribute + vecGradLossToSigmaSqr[i].elem_cal_opt(vecDistance.elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV), MATRIX_ELEM_MULT);
    }
    // Gradient loss to input
    set<feature> setGradLossToInput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setGradLossToInput[i].init(setInput[i].size());
        for(auto j=0; j<setInput[i].size(); ++j) setGradLossToInput[i][j] = setGradLossToBarX[i][j].elem_cal_opt(vecDmrSigma[j], MATRIX_ELEM_DIV) + (2 / setInput.size()) * vecGradLossToSigmaSqr[j].elem_cal_opt((setInput[i][j] - ConvBNOutput.vecMiuBeta[j]), MATRIX_ELEM_MULT) + vecGradLossToMiuBeta[j].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    }
    return setGradLossToInput;
}

vect BNGradLossToScale(set<feature> &setGradLossToOutput, ConvBN &ConvBNOutput)
{
    vect vecGradGamma(ConvBNOutput.vecMiuBeta.size(), IDX_SGL);
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
        for(auto j=0; j<setGradLossToOutput.size(); ++j)
            for(auto k=0; k<setGradLossToOutput[j][i].ELEM_CNT; ++k)
                vecGradGamma.pos_idx(i) += setGradLossToOutput[j][i].pos_idx(k) * ConvBNOutput.setBarX[j][i].pos_idx(k);
    return vecGradGamma;
}

vect BNGradLossToShift(set<feature> &setGradLossToOutput)
{
    vect vecGradBeta(setGradLossToOutput[IDX_ZERO].size(), IDX_SGL);
    for(auto i=0; i<setGradLossToOutput[IDX_ZERO].size(); ++i)
        for(auto j=0; j<setGradLossToOutput.size(); ++j)
            for(auto k=0; k<setGradLossToOutput[j][i].ELEM_CNT; ++k)
                vecGradBeta.pos_idx(i) += setGradLossToOutput[j][i].pos_idx(k);
    return vecGradBeta;
}

set<feature> BNDeduce(set<feature> &setNetInput, vect &vecBeta, vect &vecGamma, set<BN_PTR> &setbnData, uint64_t iMiniBatchSize = 0, double dEpsilon = 1e-10)
{
    auto vecEX = INSTANCE_DERIVE<ConvBN>(setbnData[0]) -> vecMiuBeta,
        vecVarX = INSTANCE_DERIVE<ConvBN>(setbnData[0]) -> vecSigmaSqr;
    for(auto i=1; i<setbnData.size(); ++i) for(auto j=0; j<INSTANCE_DERIVE<ConvBN>(setbnData[i])->vecMiuBeta.size(); ++j)
    {
        vecEX[j] += INSTANCE_DERIVE<ConvBN>(setbnData[i]) -> vecMiuBeta[j];
        vecVarX[j] += INSTANCE_DERIVE<ConvBN>(setbnData[i]) -> vecSigmaSqr[j];
    }
    if(iMiniBatchSize) for(auto i=0; i<vecEX.size(); ++i)
    {
        vecEX[i] = vecEX[i].elem_cal_opt(setbnData.size(), MATRIX_ELEM_DIV);
        vecVarX[i] = (iMiniBatchSize / (iMiniBatchSize - 1)) * vecVarX[i].elem_cal_opt(setbnData.size(), MATRIX_ELEM_DIV);
    }
    set<feature> setConvBNDeduceOutput(setNetInput.size());
    for(auto i=0; i<setNetInput.size(); ++i)
    {
        setConvBNDeduceOutput[i].init(setNetInput[i].size());
        for(auto j=0; j<setNetInput[i].size(); ++j)
        {
            auto vecBarX = (setNetInput[i][j] - vecEX[j]).elem_cal_opt(DIV_DOM(vecVarX[j], dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
            setConvBNDeduceOutput[i][j] = (vecGamma.pos_idx(j) * vecBarX).broadcast_add(vecBeta.pos_idx(j));
        }
    }
    return setConvBNDeduceOutput;
}

CONV_END