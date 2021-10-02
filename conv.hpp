NEUNET_CONV_BEGIN

vect Conv(vect &vecInput, vect &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    if(SAMP_VALID(vecInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation) &&
        SAMP_VALID(vecInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation) &&
        vecInput.is_matrix() && vecKernel.is_matrix())
    {
        auto iOutputLnCnt = SAMP_OUTPUT_DIR_CNT(vecInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation),
            iOutputColCnt = SAMP_OUTPUT_DIR_CNT(vecInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation);
        vect vecOutput(iOutputLnCnt, iOutputColCnt);
        for(auto i=0; i<iOutputLnCnt; ++i)
            for(auto j=0; j<iOutputColCnt; ++j)
                for(auto k=0; k<vecKernel.LN_CNT; ++k)
                    for(auto l=0; l<vecKernel.COL_CNT; ++l)
                        vecOutput[i][j] += vecInput[FILTER_TRACE_POS(i, k, iLnStride, iLnDilation)][FILTER_TRACE_POS(j, l, iColStride, iColDilation)] * vecKernel[k][l];
        return vecOutput;
    }
    else return blank_vect;
}

feature Conv(feature &vecInput, feature &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    auto iChannCnt = vecInput.size();
    if(iChannCnt == vecKernel.size())
    {
        feature vecOutput(iChannCnt);
        for(auto i=0; i<iChannCnt; ++i)
        {
            vecOutput[i] = Conv(vecInput[i], vecKernel[i], iLnStride, iColStride, iLnDilation, iColDilation);
            if(!vecOutput[i].is_matrix()) return blank_feature;
        }
        return vecOutput;
    }
    else return blank_feature;
}

tensor Conv(feature &vecInput, tensor &tenKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    auto iFtCnt = tenKernel.size();
    tensor tenOutput(iFtCnt);
    for(auto i=0; i<iFtCnt; ++i)
    {
        tenOutput[i] = Conv(vecInput, tenKernel[i], iLnStride, iColStride, iLnDilation, iColDilation);
        if(!tenOutput[i].size()) return blank_tensor;
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



NEUNET_CONV_END