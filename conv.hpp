CONV_BEGIN

vect Conv(vect &vecInput, vect &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    if(SAMP_VALID(vecInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation) &&
        SAMP_VALID(vecInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation) &&
        vecInput.is_matrix() && vecKernel.is_matrix())
    {
        auto vecPrepInput = vecInput.pad(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        auto iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(vecPrepInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation),
            iOutputColCnt = SAMP_OUTPUT_DIR_CNT(vecPrepInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation);
        vect vecOutput(iOutputLnCnt, iOutputColCnt);
        for(auto i=0; i<iOutputLnCnt; ++i)
            for(auto j=0; j<iOutputColCnt; ++j)
                for(auto k=0; k<vecKernel.LN_CNT; ++k)
                    for(auto l=0; l<vecKernel.COL_CNT; ++l)
                        vecOutput[i][j] += vecPrepInput[SAMP_TRACE_POS(i, k, iLnStride, iLnDilation)][SAMP_TRACE_POS(j, l, iColStride, iColDilation)] * vecKernel[k][l];
        return vecOutput;
    }
    else return blank_vect;
}

feature Conv(feature &vecInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    feature tenOutput(tenKernel.size());
    for(auto i=0; i<tenKernel.size(); ++i) for(auto j=0; j<vecInput.size(); ++j) 
    {
        if(tenOutput[i].is_matrix()) tenOutput[i] += Conv(vecInput[j], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        else tenOutput[i] = Conv(vecInput[j], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(!tenOutput[i].is_matrix()) return blank_feature;
    }
    return tenOutput;
}

vect GradLossToKernel(vect &vecGradLossToOutput, vect &vecInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto vecPrepInput = vecInput.pad(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    if(SAMP_VALID(vecPrepInput.LN_CNT, vecGradLossToOutput.LN_CNT, iLnStride, iLnDilation) &&
    SAMP_VALID(vecPrepInput.COL_CNT, vecGradLossToOutput.COL_CNT, iColStride, iColDilation)) return Conv(vecPrepInput, vecGradLossToOutput, iLnStride, iColStride, iLnDilation, iColDilation);
    else return blank_vect;
}

tensor GradLossToKernel(feature &vecGradLossToOutput, feature &vecInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    tensor tenGrad(vecGradLossToOutput.size());
    for(auto i=0; i<vecGradLossToOutput.size(); ++i)
    {
        auto iChannCnt = vecInput.size();
        tenGrad[i].init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j)
        {
            tenGrad[i][j] = GradLossToKernel(vecGradLossToOutput[i], vecInput[j], iLnStride, iColStride, iLnDilation, iColDilation);
            if(!tenGrad[i][j].is_matrix()) return blank_tensor;
        }
    }
    return tenGrad;
}

vect GradLossToInput(vect &vecGradLossToOutput, vect &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto iInputLnCnt = SAMP_INPUT_DIR_CNT(vecGradLossToOutput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation),
        iInputColCnt = SAMP_INPUT_DIR_CNT(vecGradLossToOutput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation);
    vect vecGrad(iInputLnCnt, iInputColCnt);
    for(auto i=0; i<vecGradLossToOutput.LN_CNT; ++i)
        for(auto j=0; j<vecGradLossToOutput.COL_CNT; ++j)
            for(auto k=0; k<vecKernel.LN_CNT; ++k)
                for(auto l=0; l<vecKernel.COL_CNT; ++l)
                    vecGrad[SAMP_TRACE_POS(i, k, iLnStride, iLnDilation)][SAMP_TRACE_POS(j, l, iColStride, iColDilation)] += vecGradLossToOutput[i][j] * vecKernel[k][l];
    return vecGrad.crop(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
}

feature GradLossToInput(feature &vecGradLossToOutput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    feature vecGrad;
    if(vecGradLossToOutput.size() == tenKernel.size()) for(auto i=0; i<tenKernel.size(); ++i)
    {
        if(!vecGrad.size()) vecGrad.init(tenKernel[i].size());
        for(auto j=0; j<tenKernel[i].size(); ++j)
        {
            auto vecSglGrad = GradLossToInput(vecGradLossToOutput[i], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
            if(vecSglGrad.is_matrix())
                if(vecGrad[j].is_matrix()) vecGrad[j] += vecSglGrad;
                else vecGrad[j] = vecSglGrad;
            else return blank_feature;
        }
    }
    return vecGrad;
}

vect PoolDownMaxAvg(vect &vecInput, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iPoolType = POOL_DOWN_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    vect vecOutput;
    if(SAMP_VALID(vecInput.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation) &&
    SAMP_VALID(vecInput.COL_CNT, iFilterColCnt, iColStride, iColDilation))
    {
        auto iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(vecInput.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation),
            iOutputColCnt = SAMP_OUTPUT_DIR_CNT(vecInput.COL_CNT, iFilterColCnt, iColStride, iColDilation);
        vecOutput = vect(iOutputLnCnt, iOutputColCnt);
        for(auto i=0; i<vecOutput.LN_CNT; ++i)
            for(auto j=0; j<vecOutput.COL_CNT; ++j)
            {
                double dPoolElem = 0;
                if(iPoolType == POOL_DOWN_MAX) dPoolElem = vecInput.extremum(SAMP_TRACE_POS(i, 0, iLnStride, iLnDilation), SAMP_TRACE_POS(i, iFilterLnCnt-1, iLnStride, iLnDilation), SAMP_TRACE_POS(j, 0, iColStride, iColDilation), SAMP_TRACE_POS(j, iFilterColCnt-1, iColStride, iColDilation), iLnDilation, iColDilation).val;
                else if(iPoolType == POOL_DOWN_AVG) dPoolElem = vecInput.elem_sum(SAMP_TRACE_POS(i, 0, iLnStride, iLnDilation), SAMP_TRACE_POS(i, iFilterLnCnt-1, iLnStride, iLnDilation), SAMP_TRACE_POS(j, 0, iColStride, iColDilation), SAMP_TRACE_POS(j, iFilterColCnt-1, iColStride, iColDilation), iLnDilation, iColDilation) / (iFilterLnCnt * iFilterColCnt);
                else return blank_vect;
                vecOutput[i][j] = dPoolElem;
            }
        return vecOutput;
    }
    return vecOutput;
}

vect PoolDownGlbAvg(vect &vecInput) {return vect(vecInput.elem_sum()/vecInput.ELEM_CNT);}

feature PoolDown(feature &vecInput, uint64_t iPoolType = POOL_DOWN_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    feature vecOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i)
    {
        if(iPoolType == POOL_DOWN_GAG) vecOutput[i] = PoolDownGlbAvg(vecInput[i]);
        else vecOutput[i] = PoolDownMaxAvg(vecInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iPoolType, iLnDilation, iColDilation
        );
        if(!vecOutput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

vect PoolUpGlbAvg(vect &vecInput, uint64_t iUpLnCnt, uint64_t iUpColCnt)
{
    if(vecInput.ELEM_CNT == 1)
    {
        vect vecOuput(iUpLnCnt, iUpColCnt);
        vecOuput.value_fill(vecInput.atom() / vecOuput.ELEM_CNT);
        return vecOuput;
    }
    else return blank_vect;
}

vect PoolUpMaxAvg(vect &vecInput, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, vect &vecTraceInput = vect(), uint64_t iPoolType = POOL_UP_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    auto iOutputLnCnt = 0, iOutputColnCnt = 0;
    if(vecTraceInput.is_matrix())
    {
        iOutputLnCnt = vecTraceInput.LN_CNT;
        iOutputColnCnt = vecTraceInput.COL_CNT;
    }
    else
    {
        iOutputLnCnt = SAMP_INPUT_DIR_CNT(vecInput.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation);
        iOutputColnCnt = SAMP_INPUT_DIR_CNT(vecInput.COL_CNT, iFilterColCnt, iColStride, iColDilation);
    }
    vect vecOutput(iOutputLnCnt, iOutputColnCnt);
    for(auto i=0; i<vecInput.LN_CNT; ++i) for(auto j=0; j<vecInput.COL_CNT; ++j) 
            for(auto k=0; k<iFilterLnCnt; ++k) for(auto l=0; l<iFilterColCnt; ++l)
            {
                auto iTraceLn = SAMP_TRACE_POS(i, k, iLnStride, iLnDilation),
                    iTraceCol = SAMP_TRACE_POS(j, l, iColStride, iColDilation);
                switch (iPoolType)
                {
                case POOL_UP_AVG:
                    vecOutput[iTraceLn][iTraceCol] += vecInput[i][j] / (iFilterLnCnt * iFilterColCnt);
                    break;
                case POOL_UP_MAX:
                    if(vecTraceInput[iTraceLn][iTraceCol] == vecInput[i][j]) vecOutput[iTraceLn][iTraceCol] += vecInput[i][j];
                    break;
                case POOL_UP_FIL:
                    vecOutput[iTraceLn][iTraceCol] += vecInput[i][j];
                    break;
                default: return blank_vect;
                }
            }
    return vecOutput;
}

feature PoolUp(feature &vecInput, uint64_t iPoolType = POOL_UP_MAX, feature &vecTraceInput = feature(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    feature vecOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i) 
    {
        if(iPoolType == POOL_UP_GAG) vecInput[i] = PoolUpGlbAvg(vecInput[i], vecTraceInput[i].LN_CNT, vecTraceInput[i].COL_CNT);
        else vecInput[i] = PoolUpMaxAvg(vecInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, vecTraceInput[i], iPoolType, iLnDilation, iColDilation);
        if(!vecInput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

CONV_END