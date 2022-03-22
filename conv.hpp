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
    feature vecOutput(tenKernel.size());
    for(auto i=0; i<tenKernel.size(); ++i) for(auto j=0; j<vecInput.size(); ++j) 
    {
        auto vecSglMap = Conv(vecInput[j], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
        if(vecOutput[i].is_matrix()) vecOutput[i] += vecSglMap;
        else vecOutput[i] = std::move(vecSglMap);
        if(!vecOutput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

vect GradLossToKernel(vect &vecGradLossToOutput, vect &vecInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto vecPrepInput = vecInput.pad(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    return Conv(vecPrepInput, vecGradLossToOutput, iLnStride, iColStride, iLnDilation, iColDilation);
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
                else vecGrad[j] = std::move(vecSglGrad);
            else return blank_feature;
        }
    }
    return vecGrad;
}

feature Im2ColOutputTransform(vect &vecIm2ColOutput, uint64_t iLnCnt)
{
    feature vecAns(vecIm2ColOutput.COL_CNT);
    auto iColCnt = vecIm2ColOutput.LN_CNT / iLnCnt;
    for(auto i=0; i<vecAns.size(); ++i)
    {
        vecAns[i] = vecIm2ColOutput.child(0, vecIm2ColOutput.LN_CNT-1, i, i);
        vecAns[i].shape_as(iLnCnt, iColCnt);
    }
    return vecAns;
}
vect Im2ColOutputTransform(feature &vecOutput)
{
    auto iIm2ColOutputLnCnt = vecOutput[IDX_ZERO].ELEM_CNT,
        iIm2ColOutputColCnt = vecOutput.size();
    vect vecAns(iIm2ColOutputLnCnt, iIm2ColOutputColCnt);
    for(auto i=0; i<iIm2ColOutputColCnt; ++i)
        for(auto j=0; j<iIm2ColOutputLnCnt; ++j)
            vecAns[j][i] = vecOutput[i].pos_idx(j);
    return vecAns;
}

vect Im2ColInputTransform(feature &vecInput, uint64_t &iOutputLnCnt, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    vect vecAns;
    int iOutputElemCnt = 0, iOutputColCnt, iFilterElemCnt = iFilterLnCnt * iFilterColCnt;
    for(auto i=0; i<vecInput.size(); ++i)
    {
        auto vecPrepInputChann = vecInput[i].pad(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);        
        if(!vecAns.is_matrix())
        {
            iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(vecPrepInputChann.LN_CNT, iFilterLnCnt, iLnStride, iLnDilation);
            iOutputColCnt = SAMP_OUTPUT_DIR_CNT(vecPrepInputChann.COL_CNT, iFilterColCnt, iColStride, iColDilation);
            iOutputElemCnt = iOutputLnCnt * iOutputColCnt;
            vecAns = vect(iOutputElemCnt, vecInput.size()*iFilterElemCnt);
        }
        for(auto j=0; j<iOutputElemCnt; ++j) for(auto k=0; k<iFilterElemCnt; ++k)
        {
            auto iAnsLn = j, iAnsCol = i*iFilterElemCnt+k;
            auto iOutputPos = mtx::mtx_elem_pos(j, iOutputColCnt), iFilterPos = mtx::mtx_elem_pos(k, iFilterColCnt);
            vecAns[iAnsLn][iAnsCol] = vecPrepInputChann[SAMP_TRACE_POS(iOutputPos.ln, iFilterPos.ln, iLnStride, iLnDilation)][SAMP_TRACE_POS(iOutputPos.col, iFilterPos.col, iColStride, iColDilation)];
        }
    }
    return vecAns;
}

feature Im2ColInputTransform(vect &vecInput, uint64_t iOutputLnCnt, uint64_t iFilterLnCnt, uint64_t iFilterColCnt, uint64_t iLnStride, uint64_t iColStride, bool bGradFlag = true, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    int iFilterElemCnt = iFilterLnCnt * iFilterColCnt,
        iOutputColCnt = vecInput.LN_CNT / iOutputLnCnt,
        iInputLnCnt = 0, iInputColCnt = 0;
    feature vecAns(vecInput.COL_CNT/iFilterElemCnt);
    for(auto i=0; i<vecAns.size(); ++i)
    {
        if(!iInputColCnt) iInputColCnt = SAMP_INPUT_DIR_CNT(iOutputColCnt, iFilterColCnt, iColStride, iColDilation);
        if(!iInputLnCnt) iInputLnCnt = SAMP_INPUT_DIR_CNT(iOutputLnCnt, iFilterLnCnt, iLnStride, iLnDilation);
        vect vecPrepAns(iInputLnCnt, iInputColCnt);
        for(auto j=0; j<vecInput.LN_CNT; ++j) for(auto k=0; k<iFilterElemCnt; ++k)
        {
            auto iCurrInputLn = j, iCurrInputCol = i*iFilterElemCnt+k;
            auto iOutputPos = mtx::mtx_elem_pos(j, iOutputColCnt), iFilterPos = mtx::mtx_elem_pos(k, iFilterColCnt);
            auto iTraceLn = SAMP_TRACE_POS(iOutputPos.ln, iFilterPos.ln, iLnStride, iLnDilation),
                iTraceCol = SAMP_TRACE_POS(iOutputPos.col, iFilterPos.col, iColStride, iColDilation);
            if(bGradFlag) vecPrepAns[iTraceLn][iTraceCol] += vecInput[iCurrInputLn][iCurrInputCol];
            else vecPrepAns[iTraceLn][iTraceCol] = vecInput[iCurrInputLn][iCurrInputCol];
        }
        vecAns[i] = vecPrepAns.crop(iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    }
    return vecAns;
}

feature ConvIm2Col(vect &vecIm2ColInput, vect &vecIm2ColKernel, uint64_t iOutputLnCnt)
{
    auto vecIm2ColOutput = fc::Output(vecIm2ColKernel, vecIm2ColInput);
    return Im2ColOutputTransform(vecIm2ColOutput, iOutputLnCnt);
}

feature GradLossToInputIm2Col(vect &vecIm2ColGradLossToOutput, vect &vectIm2ColKernel, uint64_t iOutputLnCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto vecIm2ColGrad = fc::GradLossToWeight(vecIm2ColGradLossToOutput, vectIm2ColKernel);
    return Im2ColInputTransform(vecIm2ColGrad, iOutputLnCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, true, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
}

vect GradLossToKernelIm2Col(vect &vecIm2ColGradLossToOutput, vect &vecIm2ColInput) { return fc::GradLossToInput(vecIm2ColGradLossToOutput, vecIm2ColInput); }

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
        else vecOutput[i] = PoolDownMaxAvg(vecInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iPoolType, iLnDilation, iColDilation);
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
        if(iPoolType==POOL_UP_AVG)
        {
            auto dElemGrad = vecInput[i][j] / (iFilterLnCnt * iFilterColCnt);
            for(auto k=0; k<iFilterLnCnt; ++k) for(auto l=0; l<iFilterColCnt; ++l) vecOutput[SAMP_TRACE_POS(i, k, iLnStride, iLnDilation)][SAMP_TRACE_POS(j, l, iColStride, iColDilation)] = dElemGrad;
        }
        else if(iPoolType==POOL_UP_MAX && vecTraceInput.is_matrix())
        {
            auto posExtrm = vecTraceInput.extremum(SAMP_TRACE_POS(i, 0, iLnStride, iLnDilation), SAMP_TRACE_POS(i, iFilterLnCnt-1, iLnStride, iLnDilation), SAMP_TRACE_POS(j, 0, iColStride, iColDilation), (j, iFilterColCnt-1, iColStride, iColDilation), iLnDilation, iColDilation);
            for(auto k=0; k<posExtrm.pos_list.size(); ++k) vecOutput[posExtrm.pos_list[k].ln][posExtrm.pos_list[k].col] += vecInput[i][j];
        }
        else return blank_vect;
    return vecOutput;
}

feature PoolUp(feature &vecInput, uint64_t iPoolType = POOL_UP_MAX, feature &vecTraceInput = feature(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    feature vecOutput(vecInput.size());
    for(auto i=0; i<vecInput.size(); ++i) 
    {
        if(iPoolType == POOL_UP_GAG) vecOutput[i] = PoolUpGlbAvg(vecInput[i], vecTraceInput[i].LN_CNT, vecTraceInput[i].COL_CNT);
        else vecOutput[i] = PoolUpMaxAvg(vecInput[i], iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, vecTraceInput[i], iPoolType, iLnDilation, iColDilation);
        if(!vecOutput[i].is_matrix()) return blank_feature;
    }
    return vecOutput;
}

CONV_END