NEUNET_CONV_BEGIN

bool FeatureValid(feature &vecInput)
{
    for(auto i=1; i<vecInput.size(); ++i)
        if(!vecInput[i].shape_valid(vecInput[i-1])) return false;
    return true;
}

uint64_t FilterBlockCnt(uint64_t iKernelDirCnt, uint64_t iDirDilation) {return (iDirDilation + 1) * iKernelDirCnt - iDirDilation;}

bool ParaValid(uint64_t iInputDirCnt, uint64_t iKernelDirCnt, uint64_t iDirStride, uint64_t iDirDilation)
{
    auto iElemDirSize = FilterBlockCnt(iKernelDirCnt, iDirDilation);
    auto iInputDirStrideSize = iInputDirCnt - iElemDirSize;
    return iInputDirStrideSize % iDirStride == 0;
}

uint64_t OutputDirCnt(uint64_t iInputDirCnt, uint64_t iKernelDirCnt, uint64_t iDirStride, uint64_t iDirDilation)
{
    auto iKernelDirBlockSize = FilterBlockCnt(iKernelDirCnt, iDirStride);
    auto iDirDist = iInputDirCnt - iKernelDirBlockSize;
    return iDirDist / iDirStride + 1;
}

uint64_t InputDirCnt(uint64_t iOutputDirCnt, uint64_t iKernelDirCnt, uint64_t iDirStride, uint64_t iDirDilation)
{
    auto nKernelDirBlockSize = FilterBlockCnt(iKernelDirCnt, iDirDilation);
    auto nDirDist = (iOutputDirCnt - 1) * iDirStride;
    return nDirDist + nKernelDirBlockSize;
}

uint64_t KernelSlideTraceOutputPosOnInput(uint64_t iOutputDir, uint64_t iFilterDir, uint64_t iDirStride, uint64_t iDirDilation) {return iOutputDir * iDirStride + iFilterDir * (1 + iDirDilation);}

vect Conv(vect &vecInput, vect &vecKernel, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
{
    if(ParaValid(vecInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation) &&
        ParaValid(vecInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation) &&
        vecInput.is_matrix() && vecKernel.is_matrix())
    {
        auto iOutputLnCnt = OutputDirCnt(vecInput.LN_CNT, vecKernel.LN_CNT, iLnStride, iLnDilation),
            iOutputColCnt = OutputDirCnt(vecInput.COL_CNT, vecKernel.COL_CNT, iColStride, iColDilation);
        vect vecOutput(iOutputLnCnt, iOutputColCnt);
        for(auto i=0; i<iOutputLnCnt; ++i)
            for(auto j=0; j<iOutputColCnt; ++j)
                for(auto k=0; k<vecKernel.LN_CNT; ++k)
                    for(auto l=0; l<vecKernel.COL_CNT; ++l)
                        vecOutput[i][j] += vecInput[KernelSlideTraceOutputPosOnInput(i, k, iLnStride, iLnDilation)][KernelSlideTraceOutputPosOnInput(j, l, iColStride, iColDilation)] * vecKernel[k][l];
        return vecOutput;
    }
    else return blank_vect;
}



NEUNET_CONV_END