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

tensor GradLossToKernel(set<feature> &setGradLossToOutput, set<feature> &setInput, uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
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

set<feature> Pool(set<feature> &vecInput, uint64_t iPoolType = POOL_DOWN_MAX, bool bDownSamp = true, mtx::matrix &vecTraceInput = mtx::matrix(), uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
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
    bagrt::net_queue<feature> setBarX;
    bagrt::net_queue<feature> setY;
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

CONV_END