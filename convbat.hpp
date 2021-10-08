CONV_BEGIN

tensor InitKernel(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
{
    tensor tenKernel(iAmt);
    for(auto i=0; i<iAmt; ++i)
    {
        tenKernel[i].init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j) tenKernel[i][j] = vect(iLnCnt, iColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
    }
    return tenKernel;
}

set<tensor> Conv(set<feature> &setInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    set<tensor> setOutput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setOutput[i] = Conv(setInput[i], tenKernel, iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!setOutput[i].size()) return blank_ten_seq;
    }
    return setOutput;
}

set<tensor> GradLossToKernelSet(set<feature> &setGradLossToOutput, set<feature> &setInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    set<tensor> setGradLossToKernel(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setGradLossToKernel[i] = GradLossToKernel(setGradLossToOutput[i], setInput[i], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!setGradLossToKernel[i].size()) return blank_ten_seq;
    }
    return setGradLossToKernel;
}

tensor GradLossToKernel(set<feature> &setGradLossToOutput, tensor &setInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto setGradLossToKernel = GradLossToKernelSet(setGradLossToOutput, setInput, iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    tensor tenGradLossToKernel;
    if(setGradLossToKernel.size())
    {
        for(auto i=0; i<setGradLossToKernel.size(); ++i)
        {
            if(!tenGradLossToKernel.size()) tenGradLossToKernel = setGradLossToKernel[i];
            else for(auto j=0; j<setGradLossToKernel[i].size(); ++j)
                for(auto k=0; k<setGradLossToKernel[i][j].size(); ++k)
                {
                    tenGradLossToKernel[j][k] += setGradLossToKernel[i][j][k];
                    if(!tenGradLossToKernel[j][k].is_matrix()) return blank_tensor;
                }
        }
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

set<feature> Pool(set<feature> &vecInput, uint64_t iPoolType = POOL_DOWN_MAX, bool bDownSamp = true, vect &vecTraceInput = vect(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    set<feature> setOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i)
    {
        if(bDownSamp) setOutput[i] = PoolDown(vecInput[i], iPoolType, iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);
        else setOutput[i] = PoolUp(vecInput[i], iPoolType, vecTraceInput, iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);
        if(!setOutput[i].size()) return blank_ft_seq;
    }
    return setOutput;
}

struct ConvBN
{
    feature vecMiuBeta;
    feature vecSigmaSqr;
    set<feature> setBarX;
    set<feature> setY;
    ConvBN(){}
    ConvBN(ConvBN &ConvBNVal) {*this = ConvBNVal;}
    ConvBN(ConvBN &&ConvBNVal) {*this = std::move(ConvBNVal);}
    void operator=(ConvBN &ConvBNVal)
    {
        vecMiuBeta = ConvBNVal.vecMiuBeta;
        vecSigmaSqr = ConvBNVal.vecSigmaSqr;
        setBarX = ConvBNVal.setBarX;
        setY = ConvBNVal.setY;
    }
    void operator=(ConvBN &&ConvBNVal)
    {
        vecMiuBeta = std::move(ConvBNVal.vecMiuBeta);
        vecSigmaSqr = std::move(ConvBNVal.vecSigmaSqr);
        setBarX = std::move(ConvBNVal.setBarX);
        setY = std::move(ConvBNVal.setY);
    }
    // ~ConvBN() {}
};

vect BNInitScaleShift(uint64_t nChannCnt, bool bConvScaleShift = CONV_BN_SCALE)
{
    vect vecSS(nChannCnt, 1);
    if(bConvScaleShift == CONV_BN_SCALE) for(auto i=0; i<nChannCnt; ++i) vecSS[i][ZERO_IDX] = 1;
    return vecSS;
}

ConvBN BNTrain(set<feature> &setInput, vect &vecBeta, vect &vecGamma, bool bGetOutput = true, double dEpsilon = 1e-5)
{
    ConvBN BNOutput;
    auto iChannCnt = setInput[ZERO_IDX].size();
    // Average & Variance
    BNOutput.vecMiuBeta.init(iChannCnt);
    BNOutput.vecSigmaSqr.init(iChannCnt);
    for(auto i=0; i<iChannCnt; ++i)
    {
        for(auto j=0; j<setInput.size(); ++j)
        {
            if(!BNOutput.vecMiuBeta[i].is_matrix()) BNOutput.vecMiuBeta[i] = vect(setInput[j][i].get_ln_cnt(), setInput[j][i].get_col_cnt());
            BNOutput.vecMiuBeta[i] += setInput[j][i];
        }
        BNOutput.vecMiuBeta[i].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
        for(auto j=0; j<setInput.size(); ++j)
        {
            if(!BNOutput.vecSigmaSqr[i].is_matrix()) BNOutput.vecSigmaSqr[i] = vect(setInput[j][i].get_ln_cnt(), setInput[j][i].get_col_cnt());
            BNOutput.vecSigmaSqr[i] += (setInput[j][i] - BNOutput.vecMiuBeta[i]).elem_cal_opt(2, MATRIX_ELEM_POW);
        }
        BNOutput.vecSigmaSqr[i].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    }
    if(bGetOutput)
    {
        // Normalize & Output
        BNOutput.setBarX.init(setInput.size());
        BNOutput.setY.init(setInput.size());
        for(auto i=0; i<setInput.size(); ++i)
        {
            BNOutput.setBarX[i].init(iChannCnt);
            for(auto j=0; j<iChannCnt; ++j) BNOutput.setBarX[i][j] = (setInput[i][j] - BNOutput.vecMiuBeta[j]).elem_cal_opt(DIV_DOM(BNOutput.vecSigmaSqr[j], dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
            BNOutput.setY[i].init(iChannCnt);
            for(auto j=0; j<iChannCnt; ++j) BNOutput.setY[i][j] = (vecGamma[j][ZERO_IDX] * BNOutput.setBarX[i][j]).broadcast_add(vecBeta[j][ZERO_IDX]);
        }
    }
    return BNOutput;
}

set<feature> BNGradLossToInput(ConvBN &ConvBNOutput, set<feature> &setInput, set<feature> &setGradLossToOutput, vect &vecGamma, double dEpsilon = 1e-5)
{
    // Sigma & Square-powered sigma
    feature vecDmrSigmaSqr(ConvBNOutput.vecMiuBeta.size()), vecDmrSigma(ConvBNOutput.vecMiuBeta.size());
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
    {
        vecDmrSigmaSqr[i] = DIV_DOM(ConvBNOutput.vecSigmaSqr[i], dEpsilon);
        vecDmrSigma[i] = vecDmrSigmaSqr[i].elem_cal_opt(0.5, MATRIX_ELEM_POW);
    }
    // Gradient loss to normalized output
    set<feature> setGradLossToNormOutput(setInput.size());
    for(auto i=0; i<setInput.size(); ++i)
    {
        setGradLossToNormOutput[i].init(ConvBNOutput.vecMiuBeta.size());
        for(auto j=0; j<ConvBNOutput.vecMiuBeta.size(); ++j) setGradLossToNormOutput[i][j] = setGradLossToOutput[i][j] * vecGamma[j][ZERO_IDX];
    }
    // Gradient loss to square-powered sigma
    feature vecGradLossToSigmaSqr(ConvBNOutput.vecMiuBeta.size());
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i) for(auto j=0; j<setInput.size(); ++j)
    {
        auto vecSglGradLossToSigmaSqr = ((-1) * setGradLossToNormOutput[j][i].elem_cal_opt((setInput[j][i] - ConvBNOutput.vecMiuBeta[i]), MATRIX_ELEM_MULT)).elem_cal_opt((2 * vecDmrSigmaSqr[i].elem_cal_opt(1.5, MATRIX_ELEM_POW)), MATRIX_ELEM_DIV);
        if(vecGradLossToSigmaSqr[i].is_matrix()) vecGradLossToSigmaSqr[i] += vecSglGradLossToSigmaSqr;
        else vecGradLossToSigmaSqr[i] = vecSglGradLossToSigmaSqr;
    }
    // Gradient loss to miubeta
    feature vecGradLossToMiuBeta(ConvBNOutput.vecMiuBeta.size());
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
    {
        vect vecDistribute;
        vect vecDistance;
        for(auto j=0; j<setInput.size(); ++j)
        {
            auto vecSglDistribute = (-1) * setGradLossToNormOutput[j][i].elem_cal_opt(vecDmrSigma[i], MATRIX_ELEM_DIV);
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
        setGradLossToInput[i].init(ConvBNOutput.vecMiuBeta.size());
        for(auto j=0; j<ConvBNOutput.vecMiuBeta.size(); ++j) setGradLossToInput[i][j] = setGradLossToNormOutput[i][j].elem_cal_opt(vecDmrSigma[j], MATRIX_ELEM_DIV) + vecGradLossToSigmaSqr[j].elem_cal_opt((setInput[i][j] - ConvBNOutput.vecMiuBeta[j]).elem_cal_opt((2.0 / setInput.size()), MATRIX_ELEM_MULT),MATRIX_ELEM_MULT) + vecGradLossToMiuBeta[j].elem_cal_opt(setInput.size(), MATRIX_ELEM_DIV);
    }
    return setGradLossToInput;
}

vect BNGradLossToScale(set<feature> &setGradLossToOutput, ConvBN &ConvBNOutput)
{
    vect vecGradGamma(ConvBNOutput.vecMiuBeta.size(), 1);
    for(auto i=0; i<ConvBNOutput.vecMiuBeta.size(); ++i)
    {
        vect vecTensorGradGamma;
        for(auto j=0; j<setGradLossToOutput.size(); ++j)
        {
            auto vecSglGradGamma = setGradLossToOutput[j][i].elem_cal_opt(ConvBNOutput.setBarX[j][i], MATRIX_ELEM_MULT);
            if(vecTensorGradGamma.is_matrix()) vecTensorGradGamma += vecSglGradGamma;
            else vecTensorGradGamma = vecSglGradGamma;
        }
        vecGradGamma[i][ZERO_IDX] = vecTensorGradGamma.elem_sum();
    }
    return vecGradGamma;
}

vect BNGradLossToShift(set<feature> &setGradLossToOutput)
{
    vect vecGradBeta(setGradLossToOutput[ZERO_IDX].size(), 1);
    for(auto i=0; i<setGradLossToOutput[ZERO_IDX].size(); ++i)
    {
        vect vecSglTensorGradBeta;
        for(auto j=0; j<setGradLossToOutput.size(); ++j)
            if(vecSglTensorGradBeta.is_matrix()) vecSglTensorGradBeta += setGradLossToOutput[j][i];
            else vecSglTensorGradBeta = setGradLossToOutput[j][i];
        vecGradBeta[i][ZERO_IDX] = vecSglTensorGradBeta.elem_sum();
    }
    return vecGradBeta;
}

vect BNUpdateScaleShiftBat(vect &vecGammaBeta, vect &vecGrad, double dLearnRate = 1e-10)
{
    if(vecGammaBeta.shape_valid(vecGrad)) return vecGammaBeta - dLearnRate * vecGrad;
    else return blank_vect;
}

set<feature> BNDeduce(set<feature> &setNetInput, vect &vecBeta, vect &vecGamma, uint64_t iMiniBatchSize = 0, double dEpsilon = 1e-10)
{
    auto iBatCnt = 0;
    if(iMiniBatchSize) iBatCnt = setNetInput.size() / iMiniBatchSize;
    else iBatCnt = 1;
    feature vecEX(setNetInput[ZERO_IDX].size()), vecVarX(setNetInput[ZERO_IDX].size());
    for(auto i=0; i<iBatCnt; ++i)
    {
        ConvBN ConvBNMiniBatOutput;
        if(iMiniBatchSize) ConvBNMiniBatOutput = BNTrain(setNetInput.sub_queue(i*iMiniBatchSize, (i+1)*iMiniBatchSize-1), vecBeta, vecGamma, false);
        else ConvBNMiniBatOutput = BNTrain(setNetInput, vecBeta, vecGamma, false);
        for(auto j=0; j<setNetInput[ZERO_IDX].size(); ++j)
        {
            if(vecEX[j].is_matrix()) vecEX[j] += ConvBNMiniBatOutput.vecMiuBeta[j];
            else vecEX[j] = ConvBNMiniBatOutput.vecMiuBeta[j];
            if(vecVarX[j].is_matrix()) vecVarX[j] += ConvBNMiniBatOutput.vecSigmaSqr[j];
            else vecVarX[j] = ConvBNMiniBatOutput.vecSigmaSqr[j];
        }
    }
    for(auto i=0; i<setNetInput[ZERO_IDX].size(); ++i)
    {
        vecEX[i] = vecEX[i].elem_cal_opt(iBatCnt, MATRIX_ELEM_DIV);
        auto iVarDistbt = 0;
        if(iMiniBatchSize > 1) iVarDistbt = iMiniBatchSize / (iMiniBatchSize - 1);
        else iVarDistbt = iBatCnt;
        vecVarX[i] = iVarDistbt * vecVarX[i].elem_cal_opt(iBatCnt, MATRIX_ELEM_DIV);
    }
    set<feature> setConvBNDeduceOutput(setNetInput.size());
    for(auto i=0; i<setNetInput.size(); ++i)
    {
        setConvBNDeduceOutput[i].init(setNetInput[ZERO_IDX].size());
        for(auto j=0; j<setNetInput[ZERO_IDX].size(); ++j)
        {
            auto vecBarX = (setNetInput[i][j] - vecEX[j]).elem_cal_opt(DIV_DOM(vecVarX[j], dEpsilon).elem_cal_opt(0.5, MATRIX_ELEM_POW), MATRIX_ELEM_DIV);
            setConvBNDeduceOutput[i][j] = (vecGamma[j][ZERO_IDX] * vecBarX).broadcast_add(vecBeta[j][ZERO_IDX]);
        }
    }
    return setConvBNDeduceOutput;
}

CONV_END