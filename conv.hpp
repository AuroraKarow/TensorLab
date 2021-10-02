NEUNET_CONV_BEGIN

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

tensor Conv(feature &vecInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
{
    auto iFtCnt = tenKernel.size();
    tensor tenOutput(iFtCnt);
    for(auto i=0; i<iFtCnt; ++i)
    {
        auto iChannCnt = vecInput.size();
        tenOutput[i].init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j) 
        {
            tenOutput[i][j] = Conv(vecInput[j], tenKernel[i][j], iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
            if(!tenOutput[i][j].is_matrix()) return blank_tensor;
        }
    }
    return tenOutput;
}

feature MergeChann(tensor &tenInput)
{
    auto iChannCnt = tenInput.size();
    feature vecFeatureMap(iChannCnt);
    for(auto i=0; i<iChannCnt; ++i)
    {
        vect vecSglChann;
        for(auto j=0; j<tenInput[i].size(); ++j)
        {
            if(vecSglChann.is_matrix()) vecSglChann += tenInput[i][j];
            else vecSglChann = tenInput[i][j];
            if(!vecSglChann.is_matrix()) return blank_feature;
        }
        vecFeatureMap[i] = vecSglChann;
    }
    return vecFeatureMap;
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
    auto iFtCnt = vecGradLossToOutput.size();
    tensor tenGrad(iFtCnt);
    for(auto i=0; i<iFtCnt; ++i)
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
    auto iFtCnt = tenKernel.size();
    if(vecGradLossToOutput.size() == iFtCnt) for(auto i=0; i<iFtCnt; ++i)
    {
        auto iChannCnt = tenKernel[i].size();
        if(!vecGrad.size()) vecGrad.init(iChannCnt);
        for(auto j=0; j<iChannCnt; ++j)
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

NEUNET_CONV_END