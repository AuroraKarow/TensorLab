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